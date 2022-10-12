"""
2022.9.26
从文件夹读取数据
提供dataloader
数据增强
将数据加载到GPU

dataset 为数据集
dataloader 为数据集加载器
"""
import PIL
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

trans = torchvision.transforms.Compose([
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.Pad([28, 10]),
    torchvision.transforms.Resize((300, 300)),
    # torchvision.transforms.RandomRotation(degrees=10),
    torchvision.transforms.RandomAffine(degrees=10, scale=[0.7, 1.0]),
    torchvision.transforms.RandomGrayscale(p=0.1),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

datasets_all = ImageFolder('D:/桌面/ocr', transform=trans)
l = len(datasets_all)
datasets_train = torch.utils.data.Subset(datasets_all, range(int(0.9 * l)))
datasets_val = torch.utils.data.Subset(datasets_all, range(int(0.1 * l)))

dataloader_train = DataLoader(datasets_train, 10, shuffle=True)
dataloader_val = DataLoader(datasets_val, 4, shuffle=True)

if __name__ == '__main__':
    dataloader_test = DataLoader(datasets_all, batch_size=1, shuffle=True)
    for imgs, labels in dataloader_test:
        img1 = imgs[0, :, :, :]
        img1 = torchvision.transforms.ToPILImage()(img1)
        img1.show()
