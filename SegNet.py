import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models 

bn_momentum = 0.1

'''
vgg16_dimensions = [(64, 64, Pooling),
			  		(128, 128, Pooling),        
           		    (256, 256, 256, Pooling),  
                    (512, 512, 512, Pooling),   
                    (512, 512, 512, Pooling)]

'''

class SegNet(nn.Module):
	def __init__(self, input_channels, label_channels):
		super(SegNet, self).__init__()

		self.input_channels = input_channels
		self.label_channels = label_channels 

		#Pretrained VGG model 
		vgg = models.vgg(pretrained=True)


# call the functions like self.conv2d()
	def conv2d(self, in_channel, out_channel, kernel_size=3, padding=1):
		layer = nn.Sequential(*[nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(out_channel, momentum=bn_momentum)])
		layer = F.relu(layer)
		return layer 

	def pool(self, input_layer, kernel_size=2, stride=2, return_indices=True):
		layer, indices = F.max_pool2d(input_layer, kernel_size=kernel_size, stride=stride, return_indices=return_indices)
		return layer, indices

	def convtranspose2d(self, in_channel, out_channel, kernel_size=3, padding=1):
		layer = nn.Sequential(*[nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(out_channel)])
		layer = F.relu(layer)
		return layer 

	def forward(self,x):
		# Stage 1 
		x11 = self.conv2d(self.input_channels, 64)
		x12 = self.conv2d(x11, 64)
		x1pool, index1 = self.pool()