import torch
import torch.nn as nn
import torch.optim as optim
from math import comb
from itertools import combinations
import numpy as np

class phase_transforms(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.phases = nn.Parameter(torch.normal(0, 1, size=(ndim,)), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.get_mat()

    def get_mat(self):
        return torch.diag(torch.exp(torch.complex(torch.zeros(self.phases.shape[0]), self.phases).type(torch.cfloat))).T

class plane_rotations(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.ndim_ = ndim
        self.rots = nn.Parameter(torch.normal(0, 1, size=(comb(ndim, 2),)), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.get_mat()

    def get_mat(self):
        mat = torch.eye(self.ndim_, dtype=torch.cfloat)
        for (i, j), theta in zip(combinations(range(self.ndim_), 2), self.rots):
            plane_rot_mat = torch.eye(self.ndim_, dtype=torch.cfloat)
            plane_rot_mat[i, i] += torch.cos(theta) -1
            plane_rot_mat[j, j] += torch.cos(theta) -1
            plane_rot_mat[i, j] += -torch.sin(theta)
            plane_rot_mat[j, i] += torch.sin(theta)
            mat = plane_rot_mat @ mat
        
        return mat.T

class vector_norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.T / torch.linalg.norm(x, axis=1)).T

class measure(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.abs() ** 2).type(torch.cfloat)

class qlayer(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.vnorm = vector_norm()
        self.rot1 = plane_rotations(ndim)
        self.phase_t = phase_transforms(ndim)
        self.rot2 = plane_rotations(ndim)
        self.m = measure()
    
    def get_mat(self) -> torch.Tensor:
        return self.rot1.get_mat() @ self.phase_t.get_mat() @ self.rot2.get_mat()
    
    def export_mat(self, file) -> None:
        np.save(file, self.get_mat().detach().numpy())

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.m(self.vnorm(x) @ self.get_mat())
        
        # x = self.vnorm(x)
        # x = self.rot1(x)
        # x = self.phase_t(x)
        # x = self.rot2(x)
        # x = self.m(x)

        # return x