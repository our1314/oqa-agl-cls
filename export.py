import torch.onnx.utils

from models.classify_net1 import classify_net1
from models.net_resnet18 import net_resnet18
from models.net_resnet18_rot import net_resnet18_rot

path = 'run/train/weights/epoch=13-train_acc=1.0.pth'
f = path.replace('.pth', '.onnx')

x = torch.randn(1, 3, 200, 200)
checkpoint = torch.load(path)

net = net_resnet18_rot()  # classify_net1()
net.load_state_dict(checkpoint['net'])
net.eval()
torch.onnx.export(net,
                  x,
                  f,
                  opset_version=10,
                  # do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  verbose='True')
print('export success!')
