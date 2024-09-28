import torch
import numpy as np

from PIL import Image
from diffusers import DDIMPipeline, DiffusionPipeline

from utils.utils import load
import torchvision.transforms as transforms

# Cargar el modelo entrenado de Hugging Face
model_path = "../ddim-no-128-1-42-2024-09-09-17:15/"  # Asegúrate de colocar la ruta correcta
pipeline = DiffusionPipeline.from_pretrained(model_path, use_safetensors=True)

transform = transforms.Compose([transforms.ToTensor()])
to_pil = transforms.Compose([transforms.ToPILImage()])

# Preparar la imagen de entrada
def preprocess_image(image_path, id, image_size=128):
    image = load(image_path, id)['Image']
    image = (image * 255) // np.max(image)
    image = Image.fromarray(image.astype(np.float32))
    image = image.resize((image_size, image_size))  # Asegúrate de que la imagen tenga el tamaño esperado
    return image

# Aplicar el modelo sobre la imagen
def generate_image(input_image, pipeline):
    with torch.no_grad():
        image = transform(input_image).unsqueeze(0)
        for timestep in range(1000):
            result_image = pipeline.unet(image, timestep=timestep)[0]
            image = result_image
    return result_image

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
    output_image = generate_image(input_image, pipeline)

    # Guardar la imagen generada
    save_image(output_image, output_image_path)

    print(f"Imagen generada guardada en {output_image_path}")

