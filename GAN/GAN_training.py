from _model import Generator, Discriminator
from _data import miniimagenetDataset

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
from pytorchcv.model_provider import get_model as ptcv_get_model

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--save", type=str, default="./G_ckpt", help="checkpoints")
parser.add_argument("--uap", type=str, default="", help="uap tensor")
parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10 / cifar100")
parser.add_argument("--gpu", type=str, default="0", help="gpu number")

args = parser.parse_args()

if args.gpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
"""
Hyperparameters
"""
BASE_DIM = 128
Z_DIM = 128
BATCH_SIZE = 64
ITERS = 30000
LOG_SEP = 1000
DATA_ROOT = "/tmp2/b07501122/.torch"
LAMBDA = 10
CRITIC_ITERS = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
show_z = torch.randn(32, Z_DIM)

"""
load uap
"""
if args.uap:
    # cifar mean / std
    dataset_std_pool = {
        "cifar10": torch.tensor([0.202, 0.199, 0.201]),
        "cifar100": torch.tensor([0.267, 0.256, 0.276])
    }
    dataset_std = dataset_std_pool[args.dataset]
    # UAP
    uap = torch.load(args.uap)
    uap = uap * dataset_std.unsqueeze(1).unsqueeze(2)
else:
    uap = torch.zeros(3, 32, 32)


"""
dataset
"""
# mean / std
mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + uap),
    transforms.Normalize(mean.tolist(), std.tolist()),
])

unNormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

dataset_pool = {
    "cifar10": torchvision.datasets.CIFAR10(DATA_ROOT, train=True, transform=transform),
    "cifar100": torchvision.datasets.CIFAR100(DATA_ROOT, train=True, transform=transform),
}
dataset = dataset_pool[args.dataset]

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

class inf_data_pool():
    def __init__(self, loader):
        self.iterator = iter(loader)
        self.loader = loader
    def next(self):
        try:
            data, _ = next(self.iterator)
        except:
            self.iterator = iter(self.loader)
            data, _ = next(self.iterator)
        return data
    
data_pool = inf_data_pool(loader)

"""
Model
"""
G = Generator(base_dim=BASE_DIM, z_dim=Z_DIM).to(device)
D = Discriminator(base_dim=BASE_DIM).to(device)

G_optim = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
D_optim = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

"""
GD function
"""
def calc_gradient_penalty(D, real_data, fake_data, Lambda, device):
    """
    real_data: [bs, 3, 32, 32]
    fake_data: [bs, 3, 32, 32]
    """
    assert real_data.shape == fake_data.shape
    bs, ch, w, h = real_data.shape
    
    alpha = torch.rand(bs).to(device)
    alpha = alpha.view(bs, 1, 1, 1).expand(bs, ch, w, h)
    
    interpo_data = alpha * real_data + ((1 - alpha) * fake_data)
    interpo_data.requires_grad = True
    
    D_interpo = D(interpo_data)
    
    gradients = autograd.grad(outputs=D_interpo, inputs=interpo_data,
                         grad_outputs=torch.ones(D_interpo.shape).to(device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(bs, -1)
    
    gradient_penalty = ((torch.linalg.norm(gradients, ord=2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty

"""
Training
"""
# training / logging
from tqdm import tqdm

for ite in tqdm(range(ITERS)):
    """
    training
    """
    G.train(), D.train()
    
    # FIXED G UPDATE D
    D_loss_log = 0
    for i in range(CRITIC_ITERS):
        
        # get real data
        real_data = data_pool.next().to(device)
        bs = real_data.shape[0]
        # get fake data
        z = torch.randn(bs, Z_DIM).to(device)
        with torch.no_grad():
            fake_data = G(z)
        
        # calc loss
        D_real = D(real_data).mean()
        D_fake = D(fake_data).mean()
        D_gp = calc_gradient_penalty(D, real_data.data, fake_data.data, LAMBDA, device)
        
        # combine loss
        D_loss = D_fake - D_real + D_gp
        
        # backward
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()
        
        # record
        D_loss_log += (D_real.item() + D_fake.item())
        
    D_loss_log /= CRITIC_ITERS
        
        
    # FIXED D UPDATE G
    # calc loss
    z = torch.randn(BATCH_SIZE, Z_DIM).to(device)
    G_loss = -D(G(z)).mean()
    
    # backward
    G_optim.zero_grad()
    G_loss.backward()
    G_optim.step()
    
    G_loss_log = G_loss.item()
    
    """
    logging
    """
#     if ite % LOG_SEP == 0:
#         G.eval()
#         with torch.no_grad():
#             show_imgs = G(show_z.to(device)).detach().cpu()
#         for i in range(show_imgs.shape[0]):
#             show_imgs[i] = unNormalize(show_imgs[i])
#         show_imgs = make_grid(show_imgs).permute(1, 2, 0)
#         plt.imshow(show_imgs)
#         plt.show()

torch.save(G.state_dict(), args.save)
