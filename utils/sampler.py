from typing import List, Optional, Tuple, Union, Any, Callable, Dict

import PIL
import torch
import torch.nn.functional as F
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt

from diffusers import DiffusionPipeline, ImagePipelineOutput, SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor


class SamplingPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
        inverse_scheduler (Optional[`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` for noise addition. Can be 'default' for standard
            random noise addition, None for no noise addition or [`DDIMInverseScheduler`].
        conditioning (Optional[`str`])
            Model conditioning type. Use 'concatenate' if model is conditional by channel-wise concatenation, 
            'dual' for dual input conditionind or None if model is not conditional.
        MIN_B (Optional[`int`])
            Minimum value within the image's intensity range for normalization
        MAX_B (Optional[`int`])
            Maximum value within the image's intensity range for normalization.
    """

    def __init__(self, unet, scheduler, inverse_scheduler='default', conditioning=None, MIN_B=-1024, MAX_B=3072):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        
        self.MIN_B=MIN_B
        self.MAX_B=MAX_B
        
        self.conditioning = conditioning
        
        self.scheduler = scheduler
        self.inverse_scheduler = inverse_scheduler
        
    def preprocess(self, image):
    
        image = image['image'].cpu().numpy()
    
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
            
        image = resize(image.astype(np.float32), (image.shape[0], 1, self.unet.config.sample_size, self.unet.config.sample_size))
        
        return torch.as_tensor(image).float().contiguous()#.unsqueeze(0).unsqueeze(0) 
        
    def postprocess(self, image):
        return image#*(self.MAX_B - self.MIN_B) + self.MIN_B
            
    def auto_corr_loss(self, hidden_states, generator=None):
        batch_size, channel, height, width = hidden_states.shape

        roll_amounts = torch.randint(
            height // 2, (batch_size,), generator=generator, device=hidden_states.device
        )

        reg_loss = 0.0
        for i in range(channel):
            # Seleccionar el canal actual
            noise = hidden_states[:, i, :, :].unsqueeze(1)  # Shape: [batch_size, 1, height, width]

            for b in range(batch_size):
                # Asegurarse de que noise[b] tenga 4 dimensiones
                noise_sample = noise[b].unsqueeze(0)  # Shape: [1, 1, height, width]

                # Desplazar en la dimensión de altura (dim=2)
                rolled_h = torch.roll(noise_sample, shifts=int(roll_amounts[b].item()), dims=2)
                reg_loss += (noise_sample * rolled_h).mean() ** 2

                # Desplazar en la dimensión de ancho (dim=3)
                rolled_w = torch.roll(noise_sample, shifts=int(roll_amounts[b].item()), dims=3)
                reg_loss += (noise_sample * rolled_w).mean() ** 2

            # Reducir resolución espacial
            if height > 8 and width > 8:
                noise = F.avg_pool2d(noise, kernel_size=2)
                height, width = noise.shape[2], noise.shape[3]
            else:
                break

        return reg_loss

    def kl_divergence(self, hidden_states):
        mean = hidden_states.mean()
        var = hidden_states.var()
        return var + mean**2 - 1 - torch.log(var + 1e-7)
        
    @torch.no_grad()
    def invert(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        lambda_auto_corr: float = 20.0,
        lambda_kl: float = 20.0,
        num_reg_steps: int = 5,
        num_auto_corr_rolls: int = 5,
    ):
        r"""
        Function used to generate inverted latents given a prompt and image.

        Args:
            image (`PIL.Image.Image`, *optional*):
                `Image`, or tensor representing an image batch which will be used for conditioning.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            lambda_auto_corr (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control auto correction
            lambda_kl (`float`, *optional*, defaults to 20.0):
                Lambda parameter to control Kullback–Leibler divergence output
            num_reg_steps (`int`, *optional*, defaults to 5):
                Number of regularization loss steps
            num_auto_corr_rolls (`int`, *optional*, defaults to 5):
                Number of auto correction roll steps

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero.Pix2PixInversionPipelineOutput`] or
            `tuple`:
            [`~pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero.Pix2PixInversionPipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is the inverted
            latents tensor and then second is the corresponding decoded image.
        """

        device = self._execution_device

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.inverse_scheduler.timesteps
        
        '''
        OVERRIDE
        '''
        #self.inverse_scheduler.timesteps = torch.arange(1, num_inference_steps)
        #timesteps = self.inverse_scheduler.timesteps

        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps - 2) as progress_bar:
            for i, t in enumerate(timesteps[1:-1]):
                latent_model_input = self.inverse_scheduler.scale_model_input(image, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t).sample

                # regularization of the noise prediction
                with torch.enable_grad():
                    for _ in range(num_reg_steps):
                        if lambda_auto_corr > 0:
                            for _ in range(num_auto_corr_rolls):
                                var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)
                                l_ac = self.auto_corr_loss(var, generator=generator)
                                l_ac.backward()

                                grad = var.grad.detach() / num_auto_corr_rolls
                                noise_pred = noise_pred - lambda_auto_corr * grad

                        if lambda_kl > 0:
                            var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)
                            l_kld = self.kl_divergence(var)
                            l_kld.backward()

                            grad = var.grad.detach()
                            noise_pred = noise_pred - lambda_kl * grad

                        noise_pred = noise_pred.detach()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latent_model_input).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        inverted_latents = latents.detach().clone()

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        # 9. Convert to PIL.
        if output_type == "pil":
            inverted_latents = inverted_latents.permute(0, 2, 3, 1).cpu().numpy()
            image = self.numpy_to_pil(inverted_latents)

        if not return_dict:
            return (inverted_latents)

        return ImagePipelineOutput(inverted_latents)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        images: torch.Tensor = None,
        num_inference_steps: int = 1000,
        num_noise_steps=None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            in_channels = self.unet.config.in_channels//2 if self.conditioning is not None else self.unet.config.in_channels
            image_shape = (batch_size, in_channels, *self.unet.config.sample_size)

        # Sample random noise if no image input
        if images is None:
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                images = randn_tensor(image_shape, generator=None, dtype=self.unet.dtype)
                noisy_images = images.to(self.device)
            else:
                noisy_images = randn_tensor(image_shape, generator=None, device=self.device, dtype=self.unet.dtype)
        else:
            images = self.preprocess(images)
            images = images.to(self.device)

            bs = images.shape[0]
 
            timesteps = torch.tensor(num_inference_steps).long()
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if isinstance(self.inverse_scheduler, SchedulerMixin):
                noisy_images = self.invert(images,
                    num_inference_steps=num_inference_steps,
                    output_type="torch").images
                    
            elif self.inverse_scheduler == "default":
                noise = torch.randn(images.shape).to(self.device)
                noisy_images = self.scheduler.add_noise(images, noise, timesteps-1)
                    
            else:
                noisy_images = images

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        '''
        OVERRIDE
        '''
        #self.scheduler.timesteps = torch.arange(timesteps-1, 0, -1)
        
        self.unet.eval()
        
        ######################33
      
        samples = 15
        cols = 5
        lg_size = 2
        
        # Compute number of rows and columns
        n_cols = cols + lg_size
        n_rows = samples // (n_cols - lg_size)
        
        log_space = np.logspace(0, 1, samples+1, base=10.0) - 1
        log_space = log_space / log_space[-1]
        sampled_values = (num_inference_steps-1) * log_space
        sampled_values = np.unique(sampled_values.astype(int))
        displace = 0
        
        if len(sampled_values) < samples and num_inference_steps >= samples:
            full_range = np.arange(0, num_inference_steps)
            missing_values = np.setdiff1d(full_range, sampled_values)
            missing_values = missing_values[:samples-len(sampled_values)+1]
            sampled_values = np.sort(np.concatenate((sampled_values, missing_values)))
            
        
        if num_inference_steps < samples:
            full_range = np.arange(0, num_inference_steps)
            displace = samples - num_inference_steps
        
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
        grid = plt.GridSpec(n_rows, n_cols, wspace=0.1, hspace=0.1)

        ######################33

        for t in self.progress_bar(self.scheduler.timesteps):
            #noisy_images = self.scheduler.scale_model_input(noisy_images, t)
            if self.conditioning == "concatenate":
                noisy_images = torch.cat((noisy_images, images), dim=1)
            
            # 1. predict noise model_output
            model_output = self.unet(noisy_images, t).sample

            # 2. compute previous image: x_t -> x_t-1
            noisy_images = self.scheduler.step(model_output, t, noisy_images).prev_sample
            print(noisy_images.shape)
            
            ######################################################################3333333333333
            if int(t) in sampled_values:
                index = np.where(sampled_values == int(t))[0]-1
                row = (index-displace) // cols
                col = (index-displace) % cols + lg_size
                
                ax = fig.add_subplot(grid[row, col])
                ax.imshow(noisy_images[0].cpu().permute(1,2,0).numpy(), cmap='gray')
                ax.axis('off')
                ax.set_title(f"Step {t+1}")
             
        row = -1
        col = -1
                   
        ax_main = fig.add_subplot(grid[:, :lg_size]) 
        ax_main.imshow(noisy_images[0].cpu().permute(1,2,0).numpy(), cmap='gray')
        ax_main.axis('off')
        ax_main.set_title(f"Final Step")
            
        plt.show()
            ##############################################################################33333

        noisy_images = (noisy_images / 2 + 0.5).clamp(0, 1) #################################################################33
        noisy_images = noisy_images.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            noisy_images = self.numpy_to_pil(noisy_images)
        
        noisy_images = self.postprocess(noisy_images)

        if not return_dict:
            return (noisy_images,)

        return ImagePipelineOutput(images=noisy_images)
        
    
