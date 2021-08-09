import torch
import torch.nn as nn
import torch.optim as optim

in_dat = torch.tensor([[1.,2.],[3.,2.],[2.,0.]])
out_dat = in_dat.clone() * 2 #torch.tensor([[1,2,3],[3,2,3],[2,2,2]])

class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.mat = torch.zeros(2) #nn.Linear(2, 2, True)
        self.mult = nn.Parameter(torch.Tensor([1.,1.]), requires_grad=True)

        # for i in range(2):
        #     self.mat[i, i] = self.mult[i]

        # print(f"{self.mat=}")

    def forward(self, x:torch.Tensor):
        # self.mat = torch.zeros((2, 2))
        # self.mat[0, 0] += self.mult[0]
        # self.mat[1, 1] += self.mult[1]

        mat = torch.zeros(self.mult.shape*2)
        for i in range(self.mult.shape[0]):
            mat[i, i] += self.mult[i]
        return x @ mat.T #self.mat.T
        # return x @ (self.mult[0] * self.mat.T)

tmodel = test_model()

print(tmodel(in_dat))

criterion = nn.MSELoss()
optimizer = optim.SGD(tmodel.parameters(), lr=0.001, momentum=0.9)

i=0
for epoch in range(2000):  # loop over the dataset multiple times

    running_loss = 0.0
    for _, data in enumerate(zip(in_dat, out_dat), 0):
        i+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = tmodel(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# print(f"{tmodel.lin1.weight=}")
# print(f"{tmodel.lin1.bias=}")
print(f"{tmodel.mult=}")

print(f"{tmodel(in_dat)=}")

# print(f"{tmodel.mat=}")

# print(f"{tmodel(in_dat) / out_dat}")