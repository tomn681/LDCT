import torch
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
from diffusers import DDIMPipeline, DiffusionPipeline, DDIMScheduler

from utils.utils import load
import torchvision.transforms as transforms

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Cargar el modelo entrenado de Hugging Face
model_path = "../ddim-no-128-1-42-2024-09-09-17:15/"  # Asegúrate de colocar la ruta correcta
pipeline = DiffusionPipeline.from_pretrained(model_path, use_safetensors=True).to(device)

transform = transforms.Compose([transforms.ToTensor()])
to_pil = transforms.Compose([transforms.ToPILImage()])

# Set up a DDIM scheduler
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

# Preparar la imagen de entrada
def preprocess_image(image_path, id, image_size=128):
    image = load(image_path, id)['Image']
    image = (image * 255) // np.max(image)
    image = Image.fromarray(image.astype(np.float32))
    image = image.resize((image_size, image_size))  # Asegúrate de que la imagen tenga el tamaño esperado
    return image

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
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipeline.unet(latent_model_input, t).sample

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipeline.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipeline.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    #images = pipeline.decode_latents(latents)
    print(latents.squeeze(0).cpu().numpy().shape)
    images = latents.squeeze(0).cpu().numpy()

    return images
    
# Guardar o visualizar la imagen generada
def save_image(output_image, output_path):
    output_image = to_pil(output_image.squeeze(0))
    output_image.save(output_path)

if __name__ == "__main__":
    # Ruta a la imagen de entrada y salida
    input_image_path = "../manifest-1648648375084/LDCT-and-Projection-data/C002/12-23-2021-NA-NA-62464/1.000000-Low Dose Images-39882/1-001.dcm"
    output_image_path = "./imagen_generada.png"

    # Preprocesar la imagen
    input_image = preprocess_image(input_image_path, input_image_path.split('/')[-1])

    # Generar una nueva imagen usando el modelo DDIM
    input_image = transform(input_image).unsqueeze(0)
    output_image = sample(start_latents=input_image.to(device), pipeline=pipeline, device=device)

    # Guardar la imagen generada
    save_image(output_image, output_image_path)

    print(f"Imagen generada guardada en {output_image_path}")
