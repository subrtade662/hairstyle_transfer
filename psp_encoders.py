import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module

# code borrowed from https://github.com/eladrich/pixel2style2pixel

from Generator import EqualLinear

class Flatten(Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
	""" A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
	return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
	if num_layers == 50:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 100:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=13),
			get_block(in_channel=128, depth=256, num_units=30),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 152:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=8),
			get_block(in_channel=128, depth=256, num_units=36),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	else:
		raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
	return blocks



class SEModule(Module):
	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()
		self.avg_pool = AdaptiveAvgPool2d(1)
		self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = ReLU(inplace=True)
		self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x


class bottleneck_IR(Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = MaxPool2d(1, stride)
		else:
			self.shortcut_layer = Sequential(
				Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				BatchNorm2d(depth)
			)
		self.res_layer = Sequential(
			BatchNorm2d(in_channel),
			Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
			Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut

class bottleneck_IR_SE(Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR_SE, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = MaxPool2d(1, stride)
		else:
			self.shortcut_layer = Sequential(
				Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				BatchNorm2d(depth)
			)
		self.res_layer = Sequential(
			BatchNorm2d(in_channel),
			Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
			PReLU(depth),
			Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
			BatchNorm2d(depth),
			SEModule(depth, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut




class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * 18, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, 18, 512)
        return x

# =============================================================================================




class MyHairEncoderGradualStyleWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(MyHairEncoderGradualStyleWPlus, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.encoder_hair = BackboneEncoderUsingLastLayerIntoWPlus(num_layers, mode=mode, opts=opts)
        self.encoder_face = BackboneEncoderUsingLastLayerIntoWPlus(num_layers, mode=mode, opts=opts)
        self.opts = opts

        out_features = 18 * 512
        in_features = out_features * 2
        self.flat = Flatten()
        self.map = Sequential(
            Linear(in_features=in_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            )


    def forward(self, face, hair):
        e_face = self.encoder_face(face)
        e_hair = self.encoder_hair(hair)

        e_face_flat = self.flat(e_face)
        e_hair_flat = self.flat(e_hair)

        e = torch.cat((e_face_flat, e_hair_flat), dim=1)


        x = self.map(e)
        x = x.view(-1, 18, 512)
        return x

    def forward_im_vec(self, face_im, hair_vec):
        e_face = self.encoder_face(face_im)
        e_hair = hair_vec

        e_face_flat = self.flat(e_face)
        e_hair_flat = self.flat(e_hair)

        e = torch.cat((e_face_flat, e_hair_flat), dim=1)


        x = self.map(e)
        x = x.view(-1, 18, 512)
        return x

    def forward_vec_im(self, face_vec, hair_im):
        e_face = face_vec
        e_hair = self.encoder_hair(hair_im)

        e_face_flat = self.flat(e_face)
        e_hair_flat = self.flat(e_hair)

        e = torch.cat((e_face_flat, e_hair_flat), dim=1)


        x = self.map(e)
        x = x.view(-1, 18, 512)
        return x

    def forward_vec_vec(self, face_vec, hair_vec):
        e_face = face_vec
        e_hair = hair_vec

        e_face_flat = self.flat(e_face)
        e_hair_flat = self.flat(e_hair)

        e = torch.cat((e_face_flat, e_hair_flat), dim=1)


        x = self.map(e)
        x = x.view(-1, 18, 512)
        return x

    def get_embedding(self, input_im, is_face):
        if is_face:
            e_out = self.encoder_face(input_im)
        else:
            e_out = self.encoder_hair(input_im)
        return e_out
