import torch
import numpy as np

from PIL import Image
from config import config
from tqdm.auto import tqdm
from diffusers import DDIMPipeline, DiffusionPipeline, DDIMScheduler

from utils.utils import load
import torchvision.transforms as transforms

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Cargar el modelo entrenado de Hugging Face
model_path = "../ddim-no-128-1-42-2024-03-11-21:54/"  # Asegúrate de colocar la ruta correcta
pipeline = DiffusionPipeline.from_pretrained(model_path, use_safetensors=True).to(device)

transform = transforms.Compose([transforms.ToTensor()])
to_pil = transforms.Compose([transforms.ToPILImage()])

# Set up a DDIM scheduler
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

# Preparar la imagen de entrada
def preprocess_image(image_path, id, image_size=512):
    image = load(image_path, id)
    slope = float(image['Metadata']['Rescale Slope'])
    intersect = float(image['Metadata']['Rescale Intercept'])
    image = image['Image']
    return image.astype(np.float32), (slope, intersect)

# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    start_latents=None,
    start_step=0,
    pipeline=pipeline,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=False,
    device=device,
):

    # Set num inference steps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipeline.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipeline.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipeline.unet(latent_model_input, t).sample

        # Normally we'd rely on the scheduler to handle the update step:
        latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

    images = latents.squeeze(0).cpu()#.numpy()

    return images

if __name__ == "__main__":
    # Ruta a la imagen de entrada y salida
    input_image_path = "../manifest-1648648375084/LDCT-and-Projection-data/C002/12-23-2021-NA-NA-62464/1.000000-Low Dose Images-39882/1-001.dcm"
    output_image_path = "./imagen_generada.png"

    # Preprocesar la imagen
    input_image, rescale = preprocess_image(input_image_path, input_image_path.split('/')[-1])
    input_image = input_image * rescale[0] + rescale[1]
    input_image = (input_image - config.image_mean)/config.image_std

    # Generar una nueva imagen usando el modelo DDIM
    input_image = transform(input_image).unsqueeze(0)
    
    import matplotlib.pyplot as plt
    #plt.imshow(input_image.squeeze(0).permute(1,2,0), cmap='Greys_r')
    #plt.show()
    
    output_image = sample(start_latents=input_image.to(device), pipeline=pipeline, device=device)
    
    plt.imshow(input_image.squeeze(0).permute(1,2,0), cmap='Greys_r')
    plt.axis('off')
    plt.savefig("imagen.png")
    
    plt.imshow(output_image.permute(1,2,0), cmap='Greys_r')
    plt.axis('off')
    plt.savefig(output_image_path)
    
    #plt.imshow(input_image.squeeze(0).permute(1,2,0)*output_image.permute(1,2,0), cmap='Greys_r')
    #plt.show()
    

    print(f"Imagen generada guardada en {output_image_path}")
