import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

# Data

CIFAR10_mean = (0.491, 0.482, 0.447)
CIFAR10_std = (0.202, 0.199, 0.201)

CIFAR100_mean = (0.507, 0.487, 0.441)
CIFAR100_std = (0.267, 0.256, 0.276)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_mean, CIFAR10_std),
    # transforms.Normalize(CIFAR100_mean, CIFAR100_std),
])

test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
# test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)
print('test set =', len(test_set))

batch_size = 8
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Testing

def epoch(model, loader, loss_fn):
    acc, loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        acc += (yp.argmax(dim=1) == y).sum().item()
        loss += loss.item() * x.shape[0]
    return acc / len(loader.dataset), loss / len(loader.dataset)

model = inceptionv4().to(device)
model.load_state_dict(torch.load('inceptionv4_cifar10.pth'))
# model.load_state_dict(torch.load('inceptionv4_cifar100.pth'))

# model = resnet152().to(device)
# model.load_state_dict(torch.load('resnet152_cifar10.pth'))
# model.load_state_dict(torch.load('resnet152_cifar100.pth'))

# model = vgg19_bn().to(device)
# model.load_state_dict(torch.load('vgg19_cifar10.pth'))
# model.load_state_dict(torch.load('vgg19_cifar100.pth'))

model.eval()
acc, loss = epoch(model, test_loader, nn.CrossEntropyLoss())
print(f'acc = {acc}, loss = {loss}')
