from math import sqrt

import parametrized_unitary_matrix as pum

# dim_names = ["x","y","z","w"]
# print("3d rotation")
# pum.generate_plane_rot_mat_SYMB(3, dim_names.index("z"), dim_names.index("x"), dim_names)

# print("4d rotation")
# pum.generate_plane_rot_mat_SYMB(4, dim_names.index("y"), dim_names.index("w"), dim_names)

# dim_names = ["x","y","z","w"]
# pum.generate_phase_mat_SYMB(3, dim_names[:-1])

from scipy.optimize import minimize
import numpy as np

intial_state = np.ones(2) * (1 / np.sqrt(2))
target_state = np.array([1, 0])

NDIM = 2
gen_mat = lambda P: pum.generate_unit_mat(NDIM, np.array(P[:-NDIM]), np.array(P[-NDIM:]))
MSE = lambda target_vec, predicted_vec: np.mean(np.abs((target_vec - predicted_vec)**2))

optim_f = lambda p: MSE(target_state, gen_mat(p) @ intial_state)
solution = minimize(optim_f, np.random.normal(0, 1, NDIM**2), method="Nelder-Mead")
print(f"{solution=}")
print(f"{solution.x=}")
print(f"{optim_f(solution.x)=}")
print(f"{gen_mat(solution.x) @ intial_state=}")
print(f"{np.abs((gen_mat(solution.x) @ intial_state)**2)=}")
print(f"{np.linalg.norm(gen_mat(solution.x) @ intial_state)=}")

# NDIM = 10
# f = lambda P: pum.generate_unit_mat(NDIM, np.array(P[:-NDIM]), np.array(P[-NDIM:]))

# p = np.random.normal(0, 1, NDIM**2)
# print(f"{p[-NDIM:].shape=}")
# print(f"{p[:-NDIM].shape=}")
# res_mat = f(np.random.normal(0, 1, NDIM**2))

# print(res_mat)
# print(np.abs(np.linalg.det(res_mat)))