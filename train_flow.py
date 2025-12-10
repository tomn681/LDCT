import os
import torch
import torch.nn.functional as F

from pathlib import Path
from config import config
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate import notebook_launcher
from models.DiffUNet2D import model as Unet2D
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import make_image_grid
from huggingface_hub import HfFolder, Repository, whoami
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.dataset import DefaultDataset
from utils.sampler import SamplingPipeline


# Flow-matching training entrypoint built to mirror train.py without modifying it.

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def build_dataloaders(cfg):
    if cfg.conditioning is not None:
        dataset = DefaultDataset("./DefaultDataset", img_size=cfg.image_size, s_cnt=cfg.slices, diff=False)
    else:
        dataset = DefaultDataset("./DefaultDataset", img_size=cfg.image_size, s_cnt=cfg.slices)

    loader_args = dict(batch_size=cfg.train_batch_size, num_workers=4, pin_memory=True)
    return torch.utils.data.DataLoader(dataset, shuffle=True, **loader_args)


def train_loop(cfg, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(cfg.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if cfg.push_to_hub:
            repo_name = get_full_repo_name(Path(cfg.output_dir).name)
            repo = Repository(cfg.output_dir, clone_from=repo_name)
        elif cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)
        accelerator.init_trackers("train_flowmatch")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(cfg.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["target"]
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]

            # Sample continuous time for flow matching in [0, 1].
            t = torch.rand(bs, device=clean_images.device)
            timesteps = (t * (noise_scheduler.config.num_train_timesteps - 1)).long()

            # Linear path from data to noise.
            x_t = (1.0 - t[:, None, None, None]) * clean_images + t[:, None, None, None] * noise

            if cfg.conditioning == "concatenate":
                model_input = torch.cat((x_t, batch["image"]), dim=1)
            else:
                model_input = x_t

            with accelerator.accumulate(model):
                # Predict velocity (noise - data) with rectified flow objective.
                model_pred = model(model_input, timesteps, return_dict=False)[0]
                target = noise - clean_images
                loss = F.mse_loss(model_pred, target)

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

        if accelerator.is_main_process:
            pipeline = SamplingPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler, conditioning=cfg.conditioning
            )
            if (epoch + 1) % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
                imgz = batch if cfg.conditioning is not None else None
                evaluate(cfg, epoch, pipeline, imgz)
            if (epoch + 1) % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
                if cfg.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(cfg.output_dir)


def evaluate(cfg, epoch, pipeline, images):
    images = pipeline(batch_size=cfg.eval_batch_size, images=images).images[: cfg.eval_batch_size]
    image_grid = make_image_grid(images, rows=cfg.eval_batch_size // 4, cols=4)

    test_dir = os.path.join(cfg.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


if __name__ == "__main__":
    # Adjust config fields at runtime to avoid collisions with DDPM runs.
    config.model_name = f"{config.model_name}_FlowMatch"
    config.output_dir = f"train/{config.model_name.lower()}-{config.mixed_precision}-{config.image_size}-{config.slices}-{config.seed}"

    model = Unet2D
    train_dataloader = build_dataloaders(config)
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=config.num_train_timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)
