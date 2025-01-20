import os
import torch
import torch.nn.functional as F

from pathlib import Path
from config import config
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate import notebook_launcher
from models.DiffUNet2D import model as Unet2D
from diffusers.utils import make_image_grid
from huggingface_hub import HfFolder, Repository, whoami
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.dataset import DefaultDataset, CombinationDataset

from utils.sampler import SamplingPipeline


#!pip install diffusers[training]

model = Unet2D

if config.conditioning is not None:
    dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices, diff=False)
    test_dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices, diff=False, train=False)

else:
    dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices)

loader_args = dict(batch_size=config.train_batch_size, num_workers=4, pin_memory=True)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

noise_scheduler = config.scheduler(num_train_timesteps = config.num_train_timesteps)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
	
def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
        
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    global_step = 0
    
    # Now you train the model
    for epoch in range(config.num_epochs):
    
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["target"]
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # If concatenation, concatenate noisy image with LDCT
            if config.conditioning == "concatenate":
                noisy_images = torch.cat((noisy_images, batch["image"]), dim=1) #batch["image"] -> clean_images
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            break
            
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = SamplingPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler,
                                        conditioning=config.conditioning)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                imgz = batch if config.conditioning is not None else None
                evaluate(config, epoch, pipeline, imgz)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)

def evaluate(config, epoch, pipeline, images):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    
    images = pipeline(
        batch_size=config.eval_batch_size,
        images = images,
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=config.eval_batch_size//4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")

    os.makedirs(test_dir, exist_ok=True)

    image_grid.save(f"{test_dir}/{epoch:04d}.png")
###########################################################


from torchvision import transforms

preprocess = transforms.Compose([
	transforms.Resize((config.image_size, config.image_size)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.5], [0.5]),
	])


if __name__ == '__main__':
	args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
	notebook_launcher(train_loop, args, num_processes=1)
	
	
