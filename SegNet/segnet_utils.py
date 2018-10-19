import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class conv_bn_relu(nn.Module):
	def __init__(self, in_channels, filters, kernel, stride, padding, bias=True, dilation=1, batch_norm = True):
		super(conv_bn_relu, self).__init__()
		convol = nn.Conv2d(int(in_channels), int(filters), kernel_size=kernel, padding=padding, stride=stride, bias=bias, dilation=dilation)
		if batch_norm:
			self.output_unit = nn.Sequential(conv_bn_relu, nn.BatchNorm2d(int(filters)), nn.ReLU(inplace=True))
		else:
			self.output_unit = nn.Sequential(conv_bn_relu, nn.ReLU(inplace=True))
	def forward(self, inputs):
		outputs = self.output_unit(inputs)
		return outputs


class Down_2conv(nn.Module):
	def __init__(self, in_size, out_size):
		super(Down_2conv, self).__init__()
		self.conv1 = conv_bn_relu.(self, in_channels=in_size, filters=out_size, kernel = 3, stride = 1, padding = 1)
		self.conv2 = conv_bn_relu(out_size, out_size, 3, 1, 1)
		self.argmax_maxpool = nn.MaxPool2d(2, 2, return_indices=True)

	def forward(self, inputs):
		output = self.conv1(inputs)
		output = self.conv2(output)
		shape_before_pool = output.size()
		output, indices = self.argmax_maxpool(output)
		return output, indices, shape_before_pool

class Down_3conv(nn.Module):
	def __init__(self, in_size, out_size):
		super(Down_3conv, self).__init__()
		self.conv1 = conv_bn_relu(out_size, out_size, 3, 1, 1)
		self.conv2 = conv_bn_relu(out_size, out_size, 3, 1, 1)
		self.conv3 = conv_bn_relu(out_size, out_size, 3, 1, 1)
		self.argmax_maxpool = nn.MaxPool2d(2,2,return_indices=True)

	def forward(self, inputs):
		output = self.conv1(inputs)
		output = self.conv2(inputs)
		output = self.conv3(inputs)
		shape_before_pool = output.size()
		output, indices = self.argmax_maxpool(output)
		return output, indices, shape_before_pool

class Up_2conv(nn.Module):
	def __init__(self, in_size, out_size):
		super(Up_2conv, self).__init__()
		self.unpool_with_indices = nn.MaxUnpool2d(kernel_size = 2, stride= 2)
		self.conv1 = conv_bn_relu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv_bn_relu(in_size, out_size, 3, 1, 1)

	def forward(self, inputs, indices, output_size):
		output = self.unpool_with_indices(input=inputs, indices=indices, output_size=output_size)
		output = self.conv1(output)
		output = self.conv2(output)
		return output 

class Up_3conv(nn.Module):
	def __init__(self, in_size, out_size):
		super(Up_3conv, self).__init__()
		self.unpool_with_indices = nn.MaxUnpool2d(kernel_size=2, stride=2)
		self.conv1 = conv_bn_relu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv_bn_relu(in_size, in_size, 3, 1, 1)
		self.conv3 = conv_bn_relu(in_size, out_size, 3, 1, 1)

	def forward(self, inputs, indices, output_size):
		output = self.unpool_with_indices(input=inputs, indices=indices, output_size=output_size)
		output = self.conv1(output)
		output = self.conv2(output)
		output = self.conv3(output)
		return output 

























