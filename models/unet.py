""" Full assembly of the parts to form the complete network """
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')

class DefaultHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DefaultHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DefaultEncoderLayer(nn.Module):

	def __init__(self, in_channels, out_channels, bias=False):
	
		super(DefaultEncoderLayer, self).__init__()

		self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
			    		    nn.BatchNorm2d(out_channels),
			    		    nn.ReLU(inplace=True),
			    		    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
			    		    nn.BatchNorm2d(out_channels),
			    		    nn.ReLU(inplace=True))
        	
	def forward(self, x):
		return self.layer(x)
		

class DefaultDecoderLayer(nn.Module):

	def __init__(self, in_channels, out_channels, bilinear=False, bias=False):
	
		super(DefaultDecoderLayer, self).__init__()
		
		self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
			    		    nn.BatchNorm2d(out_channels),
			    		    nn.ReLU(inplace=True),
			    		    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
			    		    nn.BatchNorm2d(out_channels),
			    		    nn.ReLU(inplace=True))

		self.upsample = DefaultUpsampleLayer(in_channels, out_channels, bilinear=bilinear)
			    		    
	def forward(self, x, skip):
	
		# Upsample
		x = self.upsample(x, skip)

		# Concatenate
		x = torch.cat([skip, x], dim=1)

		# Layer
		return self.layer(x)	
		

class DefaultSkipLayer(nn.Module):

	def __init__(self, in1_channels, in2_channels, bilinear=False):
		super(DefaultSkipLayer, self).__init__()
		
	def forward(self, x):
		return x
		
class DefaultUpsampleLayer(nn.Module):

	def __init__(self, in_channels, out_channels, bilinear=False, scale=None):
		super(DefaultUpsampleLayer, self).__init__()
		scale_factor = in_channels//out_channels if scale is None else scale
		self.upsample = nn.ConvTranspose2d(in_channels, 
							out_channels, 
							kernel_size=scale_factor, 
							stride=scale_factor)
		if bilinear:
		    self.upsample = nn.Upsample(scale_factor=scale_factor, 
		    					mode='bilinear', 
		    					align_corners=True)
			    		    
	def forward(self, x, skip):
		x = self.upsample(x)
		diffY = skip.size()[2] - x.size()[2]
		diffX = skip.size()[3] - x.size()[3]
		x = F.pad(x, [diffX // 2, diffX - diffX // 2,
				diffY // 2, diffY - diffY // 2])
		return x

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, 
				bilinear=False, 
				bias=False, 
				encoder_layer=None, 
				bottleneck_layer=None,
				decoder_layer=None, 
				connection_layer=None, 
				head=None):
	
		super(UNet, self).__init__()

		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		EncoderLayer = DefaultEncoderLayer
		BottleneckLayer = DefaultEncoderLayer
		DecoderLayer = DefaultDecoderLayer
		ConnectionLayer = DefaultSkipLayer
		HeadLayer = DefaultHead

		bilinear_factor = 2 if bilinear else 1

		### Encoder Layers ###

		self.maxpool = nn.MaxPool2d(2)

		self.encoder_layer_1 = EncoderLayer(n_channels, 64)
		self.encoder_layer_2 = EncoderLayer(64, 128)
		self.encoder_layer_3 = EncoderLayer(128, 256)
		self.encoder_layer_4 = EncoderLayer(256, 512)

		### Bottleneck Layers ###

		self.bottleneck = BottleneckLayer(512, 1024//bilinear_factor)

		### Skip Layers ###

		self.skip_layer_1 = ConnectionLayer(64, 128, bilinear=bilinear)
		self.skip_layer_2 = ConnectionLayer(128, 256, bilinear=bilinear)
		self.skip_layer_3 = ConnectionLayer(256, 512, bilinear=bilinear)
		self.skip_layer_4 = ConnectionLayer(512, 1024, bilinear=bilinear)

		### Decoder Layers ###

		self.decoder_layer_1 = DecoderLayer(128, 64, bilinear)
		self.decoder_layer_2 = DecoderLayer(256, 128//bilinear_factor, bilinear)
		self.decoder_layer_3 = DecoderLayer(512, 256//bilinear_factor, bilinear)
		self.decoder_layer_4 = DecoderLayer(1024, 512//bilinear_factor, bilinear)

		### Head ###
		
		self.head = HeadLayer(64, n_classes)
		
		
	def forward(self, x):
	
		### Encoder ###
		x1 = self.encoder_layer_1(x)

		x2 = self.maxpool(x1)
		x2 = self.encoder_layer_2(x2)

		x3 = self.maxpool(x2)
		x3 = self.encoder_layer_3(x3)

		x4 = self.maxpool(x3)
		x4 = self.encoder_layer_4(x4)

		### Bottleneck ###

		z1 = self.maxpool(x4)
		z1 = self.bottleneck(z1)

		### Skip Connections ###

		skip1 = self.skip_layer_1(x1)
		skip2 = self.skip_layer_2(x2)
		skip3 = self.skip_layer_3(x3)
		skip4 = self.skip_layer_4(x4)

		### Decoder ###

		y4 = self.decoder_layer_4(z1, skip4)
		y3 = self.decoder_layer_3(y4, skip3)
		y2 = self.decoder_layer_2(y3, skip2)
		y1 = self.decoder_layer_1(y2, skip1)

		### Segmentation Mapper ###

		out = self.head(y1)

		return out
		
if __name__ == '__main__':

	print('This test may take a few seconds, please wait.')

	import torch
	import numpy as np
	from PIL import Image
	from torchvision.utils import save_image
	
	scale = 1
	
	pil_img = Image.open('./maltese-portrait.jpg')
	pil_img = pil_img.resize((572, 572), Image.BICUBIC)
	img_ndarray = np.asarray(pil_img)
	img_ndarray = img_ndarray.transpose((2, 0, 1))
		
	img = torch.as_tensor(img_ndarray.copy()).float().contiguous()
	
	# La imagen es una sola, convertir a batch de tamaño 1
	img = img.unsqueeze(0)

	n_channels = 3
	n_classes = 3


	try:
		net = UNet(n_channels, n_classes, bilinear=False, bias=False)
		out = net(img)
		save_image(out, 'Non-Bilinear.jpg')
	except:
		print('Non bilinear test failed')
		
	try:
		net = UNet(n_channels, n_classes, bilinear=True, bias=False)
		out = net(img)
		save_image(out, 'Bilinear.jpg')
	except:
		print('Bilinear test failed')
	
	print('Test Completed!')

