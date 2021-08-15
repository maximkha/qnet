import torch
from torch._C import RRefType
import torch.nn as nn

import qlayer

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlay1 = qlayer.qlayer(4)
        self.qlay2 = qlayer.qlayer(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qlay1(x)
        x = self.qlay2(x)
        return x

tnet = net()

tens = torch.Tensor([[0,1,0,1],[1,0,0,1],[0,0,0,1],[1,1,0,1]]).type(torch.cfloat)
tnetout = torch.Tensor([[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1]]).type(torch.cfloat)
tnetout = (tnetout.T / tnetout.sum(axis=1)).T

print(tens)
print(f"{tnet(tens)=}")
# print(f"{tnet.qlay1.f_2(tens)=}")

print(f"{tnet(tens).sum(dim=1)=}")

with torch.no_grad():
    print(f"{torch.linalg.det(tnet.qlay1.get_mat()).abs()=}")

import torch.optim as optim

criterion = lambda x, y: (x - y).abs().mean() #nn.MSELoss() #
# optimizer = optim.SGD(tnet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adadelta(tnet.parameters(), lr=1)

i=0
for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    i+=1

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = tnet(tens)
    # print(outputs[0].real)
    # loss = criterion(outputs[0].real, labels)
    # loss = criterion(outputs[0].real, labels)

    loss = criterion(outputs, tnetout)
    # if inputs.sum()==3:
    #     print(f"{loss=}")
    #     print(f"{outputs=}")
    #     print(f"{labels=}")
    #     exit()

    loss.backward(retain_graph=True)
    optimizer.step()

    running_loss += loss.item()
    if i % 10 == 9:
        print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 10))
        running_loss = 0.0

print(f"{tnet(tens).real=}")
print(f"{tnetout.real-tnet(tens).real=}")

print(f"{tens.real=}")
print(f"{tnetout.real=}")