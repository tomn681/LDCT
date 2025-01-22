import os
import sys
import torch

sys.path.append('../')

from utils.loss import VariationalLoss

from .model import BaseModel

from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.utils import make_image_grid
from diffusers.models.autoencoders.vae import DecoderOutput
from torchvision.transforms.functional import to_pil_image

from typing import Dict, Optional, Tuple, Union

class VAE(AutoencoderKL, BaseModel): 

    def __init__(self, in_channels=1, out_channels=1):
        super(VAE, self).__init__(in_channels, out_channels)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec, commit_loss=posterior.kl())
        
    def train_model(self, train_loader, optimizer, lr_scheduler, epochs, device, config):
        self.train()
        model = self.to(device)
        
        variational_loss = VariationalLoss(device)
        global_step = 0
        
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
        model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, lr_scheduler
        )
        
        for epoch in range(epochs):
            progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            #overall_loss = 0
            for step, batch in enumerate(train_loader):
                x = batch["target"].to(device)

                with accelerator.accumulate(model):
                    output = self.forward(x)
                    loss = variational_loss(x, output)
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
                
            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    images = output.sample[:4]
                    images = [to_pil_image(image) for image in images]
                    image_grid = make_image_grid(images, rows=1, cols=len(images))
                    test_dir = os.path.join(config.output_dir, "samples")
                    os.makedirs(test_dir, exist_ok=True)
                    image_grid.save(f"{test_dir}/{epoch:04d}.png")
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    self.save_pretrained(config.output_dir, model=accelerator.unwrap_model(model))
