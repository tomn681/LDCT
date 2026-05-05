import os
import pathlib
from sample import evaluate
from utils.sampler import SamplingPipeline

from diffusers import DDIMInverseScheduler
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler

import traceback

# For DPMSolver adaptation
class DPMSolver:
    def __init__(self, order):
        self.algorithm_type = 'dpmsolver'
        self.solver_order = order
        self.__name__ = f'DPMSolver-{self.solver_order}'

device = 'cuda'

save_image_batches = 10

save_path = "./test/"

models = [("train/ddpm_concat-no-256-1-42-2025-20-01-22:42", "concatenate"),
          ("train/ddpm-no-256-1-42-2024-02-12-15:42", "default")] #DO NOT ADD LAST "/" TO PATH

timesteps = [10, 25, 50, 150, 500, 1000]

inverse_schedulers = ["DDIM_Inverse", "DDPM"]   
#inverse_schedulers = ["DDPM"]   

#schedulers = [DDIMScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler]
#schedulers = [DDPMScheduler, DPMSolver(1), DPMSolver(2), DPMSolverMultistepScheduler]
schedulers = [DDIMScheduler, DPMSolverSDEScheduler]

#modes = ['last']
modes = ['last', 'equal']

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
                if isinstance(scheduler, DPMSolver):
                    pipeline.scheduler = DPMSolverMultistepScheduler(
                        num_train_timesteps=1000,
                        beta_start=0.0001,
                        beta_end=0.02,
                        beta_schedule='linear',
                        prediction_type='epsilon',
                        final_sigmas_type='sigma_min',
                        algorithm_type=scheduler.algorithm_type,
                        solver_order=scheduler.solver_order,
                    )

                else:
                    pipeline.scheduler = scheduler(
                        num_train_timesteps=1000,
                        beta_start=0.0001,
                        beta_end=0.02,
                        beta_schedule='linear',
                        prediction_type='epsilon',
	            )
                
                for mode in modes:
                
                    for timestep in timesteps:
                        save = os.path.join(save_path, f"{model[0].split('/')[-1]}/{inv_scheduler}/{scheduler.__name__}/{mode}/{timestep}/")
                        pathlib.Path(save).mkdir(parents=True, exist_ok=True)
                        
                        print(f"Testing:\n\t{model[0].split('/')[-1]}\n\t{inv_scheduler}\n\t{scheduler.__name__}\n\t{timestep}")
                        
                        try:
                            PSNR, RMSE, SSIM, THROUGHPUT = evaluate(pipeline, path, save=save, save_image_batches=save_image_batches, batches=1, num_inference_steps=timestep, mode=mode)
                            
                            with open(save + 'results.txt','w') as file:
                                file.write("PSNR, RMSE, SSIM, THROUGHPUT\n")
                                file.write(f"{PSNR}, {RMSE}, {SSIM}, {THROUGHPUT}")
                                
                        except Exception as err:
                            print("Test Failed: ", err)
                            traceback.print_exc()
                            with open(save + 'FAILED TEST','w') as file:
                                file.write(str(err))

