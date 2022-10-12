import PIL.Image
import torch
import torchvision.models.resnet
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet101
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

class CnnNet(Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.resnet = resnet101(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        for name, param in self.resnet.named_parameters():
            if 'fc' in name == False:
                param.requires_grad = False
        self.resnet.fc = nn.Linear(2048, 36)
        self.jh = nn.Sigmoid()

        # self.linear = nn.Linear(1000,2)
        total_params = sum(p.numel() for p in self.resnet.parameters())
        print(f'原纵参数个数：{total_params}')
        total_trainable_params = sum(p.numel() for p in self.resnet.parameters() if p.requires_grad)
        print(f'可训练参数个数：{total_trainable_params}')

    def forward(self, x):
        x = self.resnet(x)
        x = self.jh(x)

        # x = self.linear(x)
        # x = F.softmax(x)
        return x


def Train(net, loss_fn, optim, dataset_train, dataset_test):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    dataloader_train = DataLoader(dataset_train, 5, True)
    dataloader_test = DataLoader(dataset_test, 2, True)

    writer = SummaryWriter('logs')
    for epoch in range(500):
        # 训练模型
        train_loss = 0
        train_acc = 0
        net.train()
        for images, labels in dataloader_train:
            images = images.to(device)
            labels = labels.to(device)

            # img = images[0]
            # img = torchvision.transforms.ToPILImage()(img)
            # img.show()

            #前向传播
            output = net(images)
            loss = loss_fn(output, labels)

            #反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()
            train_acc += ((torch.argmax(output, dim=1) == labels).sum()).item()
            writer.add_images('train_images', images, epoch, dataformats='NCHW')

        # 模型评估
        eval_loss = 0
        eval_acc = 0
        net.eval()
        for images, labels in dataloader_test:
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = loss_fn(output, labels)
            eval_loss += loss.item()
            eval_acc += ((torch.argmax(output) == labels).sum()).item()
            writer.add_images('test_images', images, epoch, dataformats='NCHW')

        #打印训练过程
        train_loss = train_loss / len(dataset_train)
        train_acc = train_acc / len(dataset_train)
        eval_loss = eval_loss / len(dataset_test)
        eval_acc = eval_acc / len(dataset_test)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        writer.add_scalar('eval_loss', eval_loss, epoch)
        writer.add_scalar('eval_acc', eval_acc, epoch)

        print(f'epoch:{epoch},  train, loss:{train_loss}, acc:{train_acc}  eval, loss:{eval_loss}, acc:{eval_acc}')


if __name__ == '__main__':

    # aa = torch.argmax(torch.tensor([1, 3, 5, 2]), keepdim=True)
    # pass
    trans = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    dataset_all = ImageFolder('D:/桌面/ocr', transform=trans)
    dataset_train = torch.utils.data.Subset(dataset_all, range(int(0.9 * len(dataset_all))))
    dataset_test = torch.utils.data.Subset(dataset_all, range(int(0.1 * len(dataset_all))))

    #网络
    net = CnnNet()
    loss_fn = nn.CrossEntropyLoss()
    # opt = torch.optim.SGD(net.parameters(), lr=0.01)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)

    Train(net, loss_fn, opt, dataset_train, dataset_test)
