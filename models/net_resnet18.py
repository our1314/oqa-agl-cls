import torch
import torchvision.models.resnet
from torch.nn import Module, Linear
from torchvision.models.resnet import resnet101, resnet50, resnet34, resnet18


class net_resnet18(Module):
    def __init__(self):
        super(net_resnet18, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = Linear(512, 38, bias=True)  # 更改全连接层
        # self.resnet.fc.add_module('softmax', torch.nn.Softmax())
        self.softmax = torch.nn.Softmax()

        print(self.resnet)  # 打印网络模型

        # for name, param in self.resnet.named_parameters():  # 除全连接层外全部冻结，不训练
        #     if 'fc' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        total_params = sum(p.numel() for p in self.resnet.parameters())
        print(f'总参数数量：{total_params}')
        total_trainable_params = sum(p.numel() for p in self.resnet.parameters() if p.requires_grad == True)
        print(f'可训练参数数量：{total_trainable_params}')

    def forward(self, x):
        x = self.resnet(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    img = torch.randn(size=(1, 3, 350, 350))
    print(img.shape)
    net = net_resnet18()

    out = net(img)
    print(f'aaa{out.shape}')
    pass
