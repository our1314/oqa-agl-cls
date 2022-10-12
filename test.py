# import torch
# import torchvision
# from PIL import Image
#
# names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#          'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.']
# img_path = "D:/桌面/ocr/K/2022-09-23_09-58-08_+0800_2378.png"
# img = Image.open(img_path)
# img.show()
# print(img)
#
# transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((300, 300)),
#                                              torchvision.transforms.ToTensor()])
# img = transforms(img)
# print(img.shape)
# img = torch.reshape(img, (1, 3, 300, 300))
# print(img.shape)
#
# model = torch.load("out/2022.09.25_20.35.19-epoch=128-loss=0.18743-acc=1.0.pth", map_location=torch.device('cpu'))
# print(model)
# model.eval()
# with torch.no_grad():
#     output = model(img)
# print(output)
# print(output.argmax(1))
# print(names[output.argmax(1)])

import torch
import torchvision
from PIL import Image

from data.data_char2 import SquarePad
from models.classify_net1 import classify_net1
from models.net_resnet18 import net_resnet18

names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.']
img_path = "C:/Users/pc/Desktop/ocr/C/2022-09-23_09-57-30_+0800_101.png"
img = Image.open(img_path)
img.show()
print(img)

transforms = torchvision.transforms.Compose([SquarePad(),
                                             torchvision.transforms.Resize(200),
                                             torchvision.transforms.ToTensor()])
img = transforms(img)
print(img.shape)
img = torch.reshape(img, (1, 3, 200, 200))
gg = torchvision.transforms.ToPILImage()(img[0, :, :, :])
gg.show()
print(img.shape)

path = 'run/train/weights/epoch=75-train_acc=1.0.pth'
checkpoint = torch.load(path)
net = net_resnet18()
# model = torch.load(checkpoint['net'])
net.load_state_dict(checkpoint['net'])
# model = torch.load("out/2022.09.25_20.01.37-epoch=39-loss=2.23615-acc=1.0.pth", map_location=torch.device('cpu'))
# net = checkpoint['net']
# net = torch.load(checkpoint['net'], map_location=torch.device('cpu'))
# print(net)
net.eval()
# with torch.no_grad():
output = net(img)
print(output)
print(output.argmax(1))
print(names[output.argmax(1)])
