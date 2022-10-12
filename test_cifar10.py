import torch
from torch import nn
from torchvision import models
from torchvision.transforms import transforms

def train(net, trainloader, testloader, optim, loss_fn):

    for epoch in range(300):
        # 训练
        train_loss = 0
        train_acc = 0
        net.train()
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            #前向传播
            output = net(images)
            loss = loss_fn(output, labels)

            #反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()
            train_acc += ((torch.argmax(output) == labels).sum()).item()
            pass
        #评估
        net.eval()
        for images, labels in testloader:
            pass

if __name__ == '__main__':
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    net = models.resnet18(pretrained=True)
    print(net)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.fc = nn.Linear(512, 10)
    net.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)
