from torch import nn

a = [1, 2, 3, 4, 5, 6]
print(a[:1])
print(a[1:])


class loss_fn:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, output, label):
        rot_out = output[:2]
        cls_out = output[2:]

        rot_label = label[:1]
        cls_label = label[1:]

        loss1 = self.loss_fn(rot_out, rot_label)
        loss2 = self.loss_fn(cls_out, cls_label)

        sum_loss = loss1 + loss2
        return sum_loss
