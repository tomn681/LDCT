import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.dataset import DefaultDataset

import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

import models.unet as unet


'''
Dictionaries
'''
loss_dict = {'MSELoss': nn.MSELoss(),
	     }
	     
model_dict = {'U-Net': unet.UNet
	     }
	     
optimizer_dict = {'RMSprop': optim.RMSprop,
			'Adam': optim.Adam,
		  }


'''
get_args Function

Retrieves the basic configuration from command line.

Inputs:
	- Command line options:
		Required:
			- file-path:		(str)	    Path to training files. See utils.dataset for more info.
		
		Optional
			- model:		(str)       Model to train. Default: 'U-Net'
			- loss:		(str)       Loss. Default: 'MSELoss'
			- epochs:		(int)       Number of training epochs. Default: 50.
			- batch-size:		(int)       Training Batch Size. Default: 32.
			- optimizer:		(str)       Optimizer. Default: 'Adam'.
			- learning-rate:	(float)     Base learning Rate. Default: 1e-5.
			- lr-scheduler:	(str)       Learning Rate Scheduler. Default: None.
			- load:		(str)       Load model from .pth file. Default: False.
			- img-size:		(int)       Image resizing values. Default: 512.
			- validation:		(float)     Validation data percentage (0-100). Default: 10.    
			- amp:			(bool)      Use Mixed Precision. Default: False.
			- bilinear:		(bool)      Use bilinear upsampling if applicable. Default: False.
			- disable-cuda:	(bool)      Disable GPU. Default: False.
			- semantic:		(str)       Semantic Segmentation. Default: True.
			- save-checkpoints	(str)       Checkpoints directory. Default: Discard Checkpoints.
		
		Note: Semantic segmentation has n_classes outputs. Non-semantic methods could present more than one 
		      object per class. Semantic = False allows the net to output as many masks as objects are in the
		      greatest mask-number image.

Outputs:
	- args object
'''
def get_args():
	parser = argparse.ArgumentParser(description='Train Segmentation Net on images and target masks')
	
	# Image Info
	parser.add_argument('--file-path', '-fp', metavar='PTH', type=str, required=True, help='Dataset Location', dest='path')
	
	# Model
	parser.add_argument('--model', '-m', metavar='M', type=str, default='U-Net',
			choices=list(model_dict.keys()),
			help='Model')	
	# Loss
	parser.add_argument('--loss', type=str, default='MSELoss', help='Loss',
			choices=list(loss_dict.keys()))

	# Training Config
	parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
	parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')

	# Optimizer Config
	parser.add_argument('--optimizer', '-o', metavar='O', type=str, default='Adam',
			choices=list(optimizer_dict.keys()),
			help='Optimizer', dest='opt')
	parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
			help='Learning rate', dest='lr')
		        
	# Pretraining Config
	parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
	
	# Dataset Config
	parser.add_argument('--img-size', '-s', type=int, default=512, help='Image resizing values. tuple(int, int)', dest='size')
	parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
			help='Percent of the data that is used as validation (0-100)')
	parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
	
	# Net Config
	parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling if applicable')
	
	# Cuda Config
	parser.add_argument('--disable-cuda', action='store_true', default=False, help='Disable GPU', dest='cpu')
	
	# Semantic Segmentation
	parser.add_argument('--semantic', action='store_true', default=False, help='Semantic Segmentation')
	
	# Save Checkpoints
	parser.add_argument('--save-checkpoints', '-c', type=str, help='Checkpoints Directory', default='./train/unet/', dest='save_checkpoint')

	return parser.parse_args()
	

def annot_max(y, ax=None):
	xmax = np.argmax(y)
	ymax = y.max()
	text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
	if not ax:
		ax=plt.gca()
	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
	arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
	kw = dict(xycoords='data',textcoords="axes fraction",
	      arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
	ax.annotate(text, xy=(xmax, ymax), xytext=(0.86, 0.4), **kw)
	
def evaluate(net, dataloader, device, criterion, force_classes=None):
	net.eval()
	num_val_batches = len(dataloader)
	dice_score = 0

	n_classes = net.n_classes if force_classes is None else force_classes

	# iterate over the validation set
	for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
		image, mask_true = batch['image'], batch['target']
		# move images and labels to correct device and type
		image = image.to(device=device, dtype=torch.float32)
		mask_true = mask_true.to(device=device, dtype=torch.float32)
		mask_true = mask_true.permute(0, 3, 1, 2).float() ############################################################

		with torch.no_grad():
			# predict the mask
			mask_pred = net(image)

			if force_classes is not None:
				mask_pred = mask_pred['out'][:, :2, :, :]

			# convert to one-hot format
			#mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
			# compute the Dice score
			dice_score += criterion(mask_pred, mask_true)
				
	net.train()

	# Fixes a potential division by zero error
	if num_val_batches == 0:
		return dice_score
	return dice_score / num_val_batches

if __name__ == '__main__':

	args = get_args()
	device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

	# Log info
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	logging.info(f'Using device {device}')

	# 1. Dataset Creation and Network Building
	dataset = DefaultDataset(file_path=args.path, img_size=args.size, train=True, transforms=None, diff=False, s_cnt=1)
	#net_info = dataset.getinfo()
	
	model = model_dict[args.model]
	net = model(n_channels=1, n_classes=1)
		
	logging.info(f'Network:\n'
		 f'\t{net.n_channels} input channels\n'
		 f'\t{net.n_classes} output channels\n'
		 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
		 
	if args.load:
		net.load_state_dict(torch.load(args.load, map_location=device))
		logging.info(f'Model loaded from {args.load}')
		
	net.to(device=device)

	# 2. Dataset Splitting and Data Loading
	n_val = int(len(dataset) * args.val/100)
	n_train = len(dataset) - n_val
	train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
	
	loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
	train_loader = DataLoader(train_set, shuffle=True, **loader_args)
	val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
	
	# 3. Initialize Training Modules
	optimizer = optimizer_dict[args.opt](net.parameters(), lr=args.lr)#, weight_decay=1e-8, momentum=0.9)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
	#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.216)#step15gamma0.1
	#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
	grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
	criterion = loss_dict[args.loss]
	
	# 4. Initialize Logging
	experiment = wandb.init(project=args.model, resume='allow', anonymous='must')
	experiment.config.update(dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
		                  val_percent=args.val, save_checkpoint=args.save_checkpoint, img_size=args.size,
		                  amp=args.amp, lr_scheduler=scheduler, optimizer=args.opt, model=args.model,
		                  bilinear=args.bilinear, semantic=args.semantic))

	logging.info(f'''Starting training:
		Model:			 {args.model}
		Epochs:                 {args.epochs}
		Batch size:             {args.batch_size}
		Learning rate:          {args.lr}
		Training size:          {n_train}
		Validation size:        {n_val}
		Checkpoints:            {args.save_checkpoint}
		Device:                 {device.type}
		Images resizing:        {args.size}
		Mixed Precision:        {args.amp}
	''')
	
	# 5. Train model
	global_step = 0
	train_score = []
	validation_score = []
	batch_validation_score = []
	learnig_rate_val = []
	
	try:

		for epoch in range(1, args.epochs+1):
			
			# Initialize Net and Epoch 
			net.train()
			epoch_loss = 0
			
			# 
			with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
				for batch in train_loader:
					images = batch['image']
					true_masks = batch['target']

					assert images.shape[1] == net.n_channels, \
					    f'Network has been defined with {net.n_channels} input channels, ' \
					    f'but loaded images have {images.shape[1]} channels. Please check that ' \
					    'the images are loaded correctly.'

					images = images.to(device=device, dtype=torch.float32)
					true_masks = true_masks.to(device=device, dtype=torch.float32)

					with torch.cuda.amp.autocast(enabled=args.amp):
						masks_pred = net(images)
						loss = criterion(masks_pred, true_masks)

					optimizer.zero_grad(set_to_none=True)
					grad_scaler.scale(loss).backward()
					grad_scaler.step(optimizer)
					grad_scaler.update()

					pbar.update(images.shape[0])
					global_step += 1
					epoch_loss += loss.item()
					experiment.log({
					    'train loss': loss.item(),
					    'step': global_step,
					    'epoch': epoch
					})
					pbar.set_postfix(**{'loss (batch)': loss.item()})

					# Evaluation round
					division_step = (n_train // (10 * args.batch_size))
					if division_step > 0:
						if global_step % division_step == 0:
							histograms = {}
							for tag, value in net.named_parameters():
								tag = tag.replace('/', '.')
								histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
								histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

							val_score = evaluate(net, val_loader, device, criterion)
							batch_validation_score.append(val_score.cpu())
							#scheduler.step(val_score)

							logging.info('Validation Dice score: {}'.format(val_score))
							experiment.log({
							    'learning rate': optimizer.param_groups[0]['lr'],
							    'validation Dice': val_score,
							    'images': wandb.Image(images[0].cpu()),
							    'masks': {
								'true': wandb.Image(true_masks[0].float().cpu()),
								'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
							    },
							    'step': global_step,
							    'epoch': epoch,
							    **histograms
							})
				val_score = evaluate(net, val_loader, device, criterion)	
				scheduler.step(val_score)		
				validation_score.append(val_score.cpu())
				trn_score = evaluate(net, train_loader, device, criterion)
				train_score.append(trn_score.cpu())
				learnig_rate_val.append(optimizer.param_groups[0]['lr'])

			if args.save_checkpoint:
				Path(args.save_checkpoint).mkdir(parents=True, exist_ok=True)
				torch.save(net.state_dict(), str(args.save_checkpoint + f'checkpoint_epoch{epoch}-{datetime.datetime.now().isoformat()}.pth'))
				logging.info(f'Checkpoint {epoch} saved!')
			
	except (KeyboardInterrupt, OSError):
		pass
	
	fig, ax = plt.subplots()

	axes = [ax, ax.twinx()]

	axes[-1].set_frame_on(True)
	axes[-1].patch.set_visible(False)

	axes[0].plot(learnig_rate_val, linestyle='-', color='lightgray')
	axes[0].set_ylabel('Learning Rate', color='lightgray')
	axes[0].tick_params(axis='y', colors='lightgray')
	axes[0].yaxis.set_label_position('right')
	axes[0].yaxis.tick_right()
	axes[0].set_yscale('log')

	validation_line, = axes[1].plot(validation_score, linestyle='-', color='Orange')
	axes[1].set_ylabel('Dice Score', color='Black')
	axes[1].tick_params(axis='y', colors='Black')
	axes[1].yaxis.set_label_position('left')
	axes[1].yaxis.tick_left()

	train_line, = axes[1].plot(train_score, linestyle='-', color='Blue')
	
	batch_x_axis = [i/10 for i in range(len(batch_validation_score))]
	batch_validation_line, = axes[1].plot(batch_x_axis, batch_validation_score, linestyle='dotted', color='Orange')

	train_line.set_label('Train')
	validation_line.set_label('Validation')

	axes[1].legend()
	axes[1].set_title('Dice Score')
	axes[0].set_xlabel('Epoch')
	axes[1].set_ylim(0.0, 1.0)
	
	annot_max(np.array(validation_score), ax=axes[1])
	
	outpath = os.path.join(args.path, 'plots/')
	if not os.path.isdir(outpath):
		os.mkdir(outpath)
	plt.savefig(outpath + args.model + '-' + str(time.time()) + '.png')

