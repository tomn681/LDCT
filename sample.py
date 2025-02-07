from typing import List, Optional, Tuple, Union

import time
import torch
from skimage.transform import resize

from utils.sampler import SamplingPipeline
import numpy as np

from diffusers import DiffusionPipeline, ImagePipelineOutput, DDIMInverseScheduler
from diffusers.utils.torch_utils import randn_tensor

from config import config
from utils.utils import load
from utils.ssim import StructuralSimilarity
from utils.dataset import DefaultDataset, CombinationDataset

from torcheval.metrics import PeakSignalNoiseRatio, MeanSquaredError, Throughput

#######################################################################
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from models.DiffUNet2D import model as Unet2D
from diffusers import DDPMScheduler, DDIMScheduler

def evaluate(pipeline, path, save=False, save_image_batches=10, batches=1, show=False, num_inference_steps=None):
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarity()
    mse = MeanSquaredError()
    throughput_metric = Throughput() #Es el tiempo total sobre el numero de imagenes procesadas (incluye batch size)

    dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices, train=False)

    torch.manual_seed(42)
    loader_args = dict(batch_size=config.eval_batch_size, num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, **loader_args)
    
    batches = len(test_dataloader) + batches + 1 if batches < 0 else batches
    
    for idx, batch in enumerate(test_dataloader):
    
        if idx >= batches:
            break
    
        start_time = time.time()
        num_inference_steps = num_inference_steps if num_inference_steps else config.num_inference_steps
        output = pipeline(num_inference_steps=num_inference_steps, num_noise_steps=None, batch_size=config.eval_batch_size, 
                            output_type='np.array', images=batch).images#.squeeze()

        output = torch.from_numpy(output).permute(0,3,1,2)
        end_time = time.time()

        input_img = pipeline.preprocess(batch, cls='target')#.squeeze()
        
        elapsed_time = end_time - start_time
        throughput_metric.update(num_processed=batch['image'].shape[0], elapsed_time_sec=elapsed_time)
        
        for img, out in zip(input_img, output):
            psnr.update(img, out)
            ssim.update(img, out)
            mse.update(img.flatten(), out.flatten())
        
        if idx % save_image_batches == 0 and save or show:
            save = save+f"_batch_{idx}.png" if save else save
            plot_input_output_batches(batch["image"].numpy(), output.numpy(), save=save, show=show)
        
    PSNR = psnr.compute()
    SSIM = ssim.compute()
    RMSE = np.sqrt(mse.compute())
    THROUGHPUT = throughput_metric.compute()

    return PSNR, RMSE, SSIM, THROUGHPUT

def plot_input_output_batches(input_batch, output_batch, save=False, show=False):
    """
    Plots input and output images in a grid format.

    Args:
        input_batch (np.ndarray): Batch of input images (B, 1, W, H) or (W, H) if a single image.
        output_batch (np.ndarray): Batch of output images (B, 1, W, H) or (W, H) if a single image.
    """
    # Ensure inputs are in the correct shape
    if input_batch.ndim == 2:
        input_batch = input_batch[np.newaxis, np.newaxis, ...]
    elif input_batch.ndim == 3:
        input_batch = input_batch[:, np.newaxis, ...]
    
    if output_batch.ndim == 2:
        output_batch = output_batch[np.newaxis, np.newaxis, ...]
    elif output_batch.ndim == 3:
        output_batch = output_batch[:, np.newaxis, ...]
    
    B = input_batch.shape[0]  # Batch size
    max_columns = 2
    images_per_column = 2  # Input and output per "column"
    
    # Compute number of rows and columns
    n_rows = (B + max_columns - 1) // max_columns  # Ceiling division for rows
    n_cols = max_columns * images_per_column if B > 1 else 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).reshape(n_rows, n_cols)  # Ensure consistent shape

    # Plot images
    for i in range(B):
        row = i // max_columns
        col = (i % max_columns) * images_per_column
        
        # Plot input image
        axes[row, col].imshow(input_batch[i, 0], cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Input {i+1}")
        
        # Plot output image
        axes[row, col+1].imshow(output_batch[i, 0], cmap='gray')
        axes[row, col+1].axis('off')
        axes[row, col+1].set_title(f"Output {i+1}")
    
    # Hide unused subplots
    for ax in axes.flatten():
        if not ax.has_data():
            ax.axis('off')

    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        
    if show:
        plt.show()
        
    return

if __name__ == '__main__':
   
    device = 'cuda'
    
    #model_path = "../ddpm-no-256-1-42-2024-02-12-15:42" #"../ddim-no-256-1-42-2024-02-12-15:43"
    #model_path = "../ddpm-no-512-1-42-2024-12-12-21:35"
    model_path = "../ddpm_concat-no-256-1-42-2025-20-01-22:42"
    
    pipeline = SamplingPipeline.from_pretrained(model_path, use_safetensors=True, conditioning=config.conditioning).to(device)
    
    #pipeline.inverse_scheduler = "default"
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_path+"/scheduler/")
    
    
    pipeline.scheduler = config.scheduler.from_pretrained(model_path+"/scheduler/")
    
    path = "./DefaultDataset/test.txt"
    print(evaluate(pipeline, path, show=True))
    
    #input_image_path = "../manifest-1648648375084/LDCT-and-Projection-data/C002/12-23-2021-NA-NA-62464/1.000000-Low Dose Images-39882/1-001.dcm"
    
    #images = load(input_image_path, id=0)
    
    #output = pipeline(num_inference_steps=config.num_inference_steps, num_noise_steps=None, batch_size=1, output_type='np.array', images=images).images.squeeze()
    
    #plot_input_output_batches(images['Image'], output)
