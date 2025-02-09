import time

from dataclasses import dataclass

from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler#, EDMEulerScheduler
from diffusers import DDPMPipeline, DDIMPipeline

@dataclass
class TrainingConfig:
	seed = 42
	
	image_size = 256  # the generated image resolution
	
	train_batch_size = 16
	eval_batch_size = 4#16  # how many images to sample during evaluation
	
	num_epochs = 500
	num_train_timesteps = 1000
	num_inference_steps = 1000
	
	model_name = "DDPM_Concat"
	scheduler = DDPMScheduler#EDMEulerScheduler
	pipeline = DDPMPipeline #DDPM as Default for most schedulers
	
	conditioning = "concatenate" #"concatenate", None, "dual"
	
	slices = 1
	channels = 1
	
	learning_rate = 1e-4
	lr_warmup_steps = 500
	
	save_image_epochs = 1 #10
	save_model_epochs = 1 #30
	
	mixed_precision = "no" #"fp16"  # `no` for float32, `fp16` for automatic mixed precision
	
	gradient_accumulation_steps = 1
	
	push_to_hub = False  # whether to upload the saved model to the HF Hub
	hub_private_repo = False
	overwrite_output_dir = False  # overwrite the old model when re-running the notebook
	
	output_dir = f"train/{model_name.lower()}-{mixed_precision}-{image_size}-{slices}-{seed}-{time.strftime('%Y-%d-%m-%H:%M', time.localtime(time.time()))}"  # the model name locally and on the HF Hub
	
config = TrainingConfig()
