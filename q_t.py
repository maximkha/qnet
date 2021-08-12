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
print(tens)
print(f"{tnet(tens)=}")
# print(f"{tnet.qlay1.f_2(tens)=}")

print(f"{tnet(tens).sum(dim=1)=}")

with torch.no_grad():
    print(f"{torch.linalg.det(tnet.qlay1.get_mat()).abs()=}")
# print(f"{tnet.qlay1.f_2(tens).sum(dim=1)=}")

# print(f"{tnet(tens).real-tnet.qlay1.f_2(tens).real=}")