from typing import List, Optional, Tuple, Union

import torch
from skimage.transform import resize

import numpy as np

from diffusers import DiffusionPipeline, ImagePipelineOutput
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
    """

    def __init__(self, unet, scheduler, MIN_B=-1024, MAX_B=3072):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        
        self.MIN_B=MIN_B
        self.MAX_B=MAX_B
        
    def preprocess(self, image, slope=1.0, intercept=-1024):#ESTO SOLO FUNCIONA SI BATCH=1!!!!!!!
    
        image = image['image'].numpy()
    
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
    
        #slope = float(image['Metadata']['Rescale Slope'])
        #intersept = float(image['Metadata']['Rescale Intercept'])

        image = image * slope + intercept
        
        image = (image - self.MIN_B)/(self.MAX_B - self.MIN_B)
        image = resize(image.astype(np.float32), (image.shape[0], 1, self.unet.config.sample_size, self.unet.config.sample_size))
        
        return torch.as_tensor(image).float().contiguous()#.unsqueeze(0).unsqueeze(0) 
        
    def postprocess(self, image):
        return image*(self.MAX_B - self.MIN_B) + self.MIN_B

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
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        # Sample random noise if no image input
        if images is None:
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                images = randn_tensor(image_shape, generator=None, dtype=self.unet.dtype)
                images = images.to(self.device)
            else:
                images = randn_tensor(image_shape, generator=None, device=self.device, dtype=self.unet.dtype)
        else:
            images = self.preprocess(images)
            images = images.to(self.device)

            # Sample noise to add to the images
            noise = torch.randn(images.shape).to(self.device)
            bs = images.shape[0]
            
            num_noise_steps = num_noise_steps if num_noise_steps else num_inference_steps
            timesteps = torch.tensor(num_noise_steps-1).long()
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            images = self.scheduler.add_noise(images, noise, timesteps)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        self.unet.eval()

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(images, t).sample

            # 2. compute previous image: x_t -> x_t-1
            images = self.scheduler.step(model_output, t, images, generator=None).prev_sample

        images = (images / 2 + 0.5).clamp(0, 1) #################################################################33
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            images = self.numpy_to_pil(images)
        
        images = self.postprocess(images)

        if not return_dict:
            return (images,)

        return ImagePipelineOutput(images=images)
        
    
