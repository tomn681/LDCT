import os
import pathlib
from sample import evaluate
from utils.sampler import SamplingPipeline

from diffusers import DDIMInverseScheduler
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler

device = 'cuda'

save_image_batches = 10

save_path = "./test/"

models = [("../ddpm_concat-no-256-1-42-2025-20-01-22:42", "concatenate"),
          ("../ddpm-no-256-1-42-2024-02-12-15:42", "default")] #DO NOT ADD LAST "/" TO PATH

timesteps = [10, 25, 50, 150, 500, 1000]

#inverse_schedulers = ["DDIM_Inverse", "DDPM"]   
inverse_schedulers = ["DDPM"]   

#schedulers = [DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler]
schedulers = [DPMSolverMultistepScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler]

path = "./DefaultDataset/test.txt"

if __name__ == '__main__':
    for model in models:
        pipeline = SamplingPipeline.from_pretrained(model[0], use_safetensors=True, conditioning=model[1]).to(device)     
        
        for inv_scheduler in inverse_schedulers:
            pipeline.inverse_scheduler = inv_scheduler
            
            if inv_scheduler == "DDIM_Inverse":
                pipeline.inverse_scheduler = DDIMInverseScheduler.from_pretrained(model[0]+"/scheduler/")
                
            if inv_scheduler == "DDPM":    
                pipeline.ddpm_scheduler = DDPMScheduler.from_pretrained(model[0]+"/scheduler/")
                
            for scheduler in schedulers:
                pipeline.scheduler = scheduler.from_pretrained(model[0]+"/scheduler/")
                
                for timestep in timesteps:
                    save = os.path.join(save_path, f"{model[0].split('/')[-1]}/{inv_scheduler}/{scheduler.__name__}/{timestep}/")
                    pathlib.Path(save).mkdir(parents=True, exist_ok=True)
                    
                    print(f"Testing:\n\t{model[0].split('/')[-1]}\n\t{inv_scheduler}\n\t{scheduler.__name__}\n\t{timestep}")
                    
                    try:
                        PSNR, RMSE, SSIM, THROUGHPUT = evaluate(pipeline, path, save=save, save_image_batches=save_image_batches, batches=1, num_inference_steps=timestep)
                        
                        with open(save + 'results.txt','w') as file:
                            file.write("PSNR, RMSE, SSIM, THROUGHPUT\n")
                            file.write(f"{PSNR}, {RMSE}, {SSIM}, {THROUGHPUT}")
                            
                    except Exception as err:
                        print("Test Failed")
                        with open(save + 'FAILED TEST','w') as file:
                            file.write(str(e))

