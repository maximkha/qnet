from math import sqrt
import numpy as np
from numpy.linalg import det, norm
import parametrized_unitary_matrix as pum
import functools

# QBITS = 1
# NDIM = 2**QBITS
NDIM = 3

# # initialized so that each state has equal likelyhood
# state_vec = np.ones(NDIM, dtype=complex) / np.sqrt(NDIM)
state_vec = np.array([[0,1,1],[1,0,1],[0,0,1],[1,1,1]], dtype=complex)
state_vec = np.array([v/norm(v) for v in state_vec], dtype=complex) #norm vecs
state_vec = state_vec**2

print(f"{[np.abs(v**2) for v in state_vec]=}")
print(f"{[norm(v) for v in state_vec]=}")
# np.abs((np.sqrt(val) @ cur)**2)
print(f"{[np.abs((np.sqrt(v))**2) for v in state_vec]=}")

# [N*2:(N+1)*2]
layer_mat = lambda P, lay_i: pum.generate_unit_mat(NDIM, P[lay_i*NDIM*NDIM:(lay_i + 1)*NDIM*NDIM][:-NDIM], P[lay_i*NDIM*NDIM:(lay_i + 1)*NDIM*NDIM][-NDIM:])
#forward = lambda x, P, lay_N: functools.reduce(lambda cur, val: val @ cur, [x] + [layer_mat(P, lay_i) for lay_i in range(lay_N)], np.eye(NDIM))
#forward = lambda x, P, lay_N: functools.reduce(lambda cur, val: np.abs((np.sqrt(val @ cur)**2), [x] + [layer_mat(P, lay_i) for lay_i in range(lay_N)], np.eye(NDIM))
def forward(x, P, lay_N):
    x_ = x.copy()
    for i in range(lay_N):
        x_ = np.abs((np.sqrt(x_) @ layer_mat(P, i))**2)
    return x_

multi_forward = lambda xs, P, lay_N: np.array([forward(x, P, lay_N) for x in xs], dtype=complex)

mse = lambda target, value: np.mean(np.abs((target - value)**2))

N_LAY = 11 #15#12#11
params = np.random.normal(0, 1, N_LAY*NDIM**2)

print(f"{[np.abs(det(layer_mat(params, i))) for i in range(N_LAY)]=}")

# out = np.zeros(NDIM, dtype=complex)
# out[2] = 1.
# out = np.array([[0,0,1],[1,0,1],[1,0,1],[0,0,1]], dtype=complex)
out = np.array([[1,0,1],[1,0,1],[0,0,1],[0,0,1]], dtype=complex)
out = np.array([v/norm(v) for v in out]) #norm vecs
optim = lambda P: mse(out, multi_forward(state_vec, P, N_LAY))

print(f"{state_vec[0]=}")
print(f"{state_vec=}")
print(f"{forward(state_vec[0], params, N_LAY)=}")
print(f"{optim(params)=}")

from scipy import optimize

# BFGS         - 0.014427222305500002
# Powell       - 0.014515694739907515
# Nedler Mead  - 0.015187190741992
# CG           - 0.031116364110082655
# L-BFGS-B     - 0.014478052933251368
# TNC          - 0.015200123861566028
# SLSQP        - 0.03507648550800287
# COBYLA       - 0.015430103762896991
# trust-constr - 0.014471047485413863


solution = optimize.minimize(optim, params, method="L-BFGS-B")
print(f"{solution=}")
print(f"{solution.x=}")
print(f"{optim(solution.x)=}")
print(f"{multi_forward(state_vec, solution.x, N_LAY)=}")
print(f"{multi_forward(state_vec, solution.x, N_LAY).sum(axis=1)=}")

print(f"{[np.abs(det(layer_mat(solution.x, i))) for i in range(N_LAY)]=}")

#BUG: something about multi forward is broken
# too tired to figure out

# ~ max @ 1:48am

# fixed!
# ~ max @ 11:36 am

# a = np.abs((np.sqrt(state_vec[-1]) @ layer_mat(solution.x, 0))**2)
# print(f"{a=}")
# print(f"{a.sum()=}")
# b = np.abs((np.sqrt(a) @ layer_mat(solution.x, 1))**2)
# print(f"{b=}")
# print(f"{b.sum()=}")

# a = forward(state_vec[-1], solution.x, 1)
# print(f"{a=}")
# print(f"{a.sum()=}")

# a = forward(state_vec[-1], solution.x, N_LAY)
# print(f"{a=}")
# print(f"{a.sum()=}")

# print(f"{gen_mat(solution.x) @ state_vec=}")
# print(f"{np.abs((gen_mat(solution.x) @ state_vec)**2)=}")
# print(f"{np.linalg.norm(gen_mat(solution.x) @ state_vec)=}")

# now plot the function
grids = np.mgrid[0:1.01:0.01, 0:1.01:0.01]
grid_x, grid_y = grids
vecs = grids.reshape(2,-1).T
inputs = np.append(vecs, np.ones((vecs.shape[0], 1)), 1)
inputs = np.array([v/norm(v) for v in inputs]) #norm vecs
# print(inputs)
# outs = multi_forward(inputs, solution.x, N_LAY)[:,0].real
# outs = multi_forward(inputs, solution.x, N_LAY).real[:, [0,1]]
# import vec2p as v2p
# # # np.array([v/norm(v) for v in out])
# outs = v2p.states_p(outs, np.array([v/norm(v) for v in np.array([[1,1],[0,1]])]), .01)[:, 0]
# outs /= np.sqrt(2)/2

print(f"{vecs[:, 0].shape=}")
print(f"{vecs[:, 1].shape=}")
print(f"{outs.shape=}")

print(f"{grid_x.shape=}")
print(f"{grid_y.shape=}")
print(f"{outs.shape=}")

# import matplotlib.pyplot as plt
# ax = plt.axes(projection='3d')
# # print(outs)
# ax.plot_trisurf(vecs[:,0], vecs[:,1], outs)
# plt.show()
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(x=grid_x, y=grid_y, z=outs.reshape(101, 101), colorbar_x=-0.07)])
fig.show()