import torch
import torch.nn as nn

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Generator(nn.Module):
	def __init__(self, base_dim=128, z_dim=128):
		super().__init__()
		preprocess = nn.Sequential(
			nn.Linear(z_dim, 4 * 4 * 4 * base_dim),
			nn.BatchNorm1d(4 * 4 * 4 * base_dim),
			nn.LeakyReLU(0.2, inplace=True)
		)

		block1 = nn.Sequential(
			nn.ConvTranspose2d(4 * base_dim, 2 * base_dim, 2, stride=2),
			nn.BatchNorm2d(2 * base_dim),
			nn.LeakyReLU(0.2, inplace=True)
		)

		block2 = nn.Sequential(
			nn.ConvTranspose2d(2 * base_dim, base_dim, 2, stride=2),
			nn.BatchNorm2d(base_dim),
			nn.LeakyReLU(0.2, inplace=True)
		)
		convT_out = nn.ConvTranspose2d(base_dim, 3, 2, stride=2)

		self.base_dim = base_dim
		self.preprocess = preprocess
		self.block1 = block1
		self.block2 = block2
		self.convT_out = convT_out
		self.tanh = nn.Tanh()
		
		self.apply(weights_init)

	def forward(self, input):
		output = self.preprocess(input)
		output = output.view(-1, 4 * self.base_dim, 4, 4)
		output = self.block1(output)
		output = self.block2(output)
		output = self.convT_out(output)
		output = self.tanh(output)
		return output

class Discriminator(nn.Module):
	def __init__(self, base_dim=64):
		super().__init__()
		conv = nn.Sequential(
			nn.Conv2d(3, base_dim, 3, 2, padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(base_dim, 2 * base_dim, 3, 2, padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(2 * base_dim, 4 * base_dim, 3, 2, padding=1),
			nn.LeakyReLU(0.2, inplace=True)
		)
		fc = nn.Linear(4 * 4 * 4 * base_dim, 1)

		self.base_dim = base_dim
		self.conv = conv
		self.fc = fc

		self.apply(weights_init)

	def forward(self, input):
		output = self.conv(input)
		output = output.view(-1, 4 * 4 * 4 * self.base_dim)
		output = self.fc(output)
		return output
