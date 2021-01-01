CIFAR10_mean = (0.507, 0.487, 0.441)
CIFAR10_std = (0.267, 0.256, 0.276)

CIFAR100_mean = (0.507, 0.487, 0.441)
CIFAR100_std = (0.267, 0.256, 0.276)

transform = transforms.Compose([
    transforms.Lambda(lambda x: Image.open(x)),
    transforms.ToTensor(),
	transforms.Normalize(MEAN, STD),
])

class ImgDataset(Dataset):
    def __init__(self, path, transform=transform):
        """
        path/XXX.png
        """
        self.data = []
        self.label = []
        for im in glob.glob(f'{path}/*'):
            self.data.append(im)
            self.label.append(int(im.split('/')[-1].split('.')[0].split('_')[1]))

        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]

    def __len__(self):
        return len(self.data)
