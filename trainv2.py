import os
import torch
#import torch.nn.functional as F

#from pathlib import Path
from config import config
#from tqdm.auto import tqdm
#from accelerate import Accelerator
#from accelerate import notebook_launcher
#from models.DiffUNet2D import model as Unet2D
from models.vae import VAE
#from diffusers.utils import make_image_grid
#from huggingface_hub import HfFolder, Repository, whoami
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.dataset import DefaultDataset, CombinationDataset


#!pip install diffusers[training]

model = VAE()

dataset = DefaultDataset('./DefaultDataset', img_size=config.image_size, s_cnt=config.slices)

loader_args = dict(batch_size=config.train_batch_size, num_workers=4, pin_memory=True)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

#noise_scheduler = config.scheduler(num_train_timesteps = config.num_train_timesteps)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

#model.train_model(optimizer, config.num_epochs, "cuda")

print(model)
print(model.train_model(train_dataloader, optimizer, lr_scheduler, 1, "cuda", config))
