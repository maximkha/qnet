import numpy as np
from math import comb
from itertools import combinations

def generate_plane_rot_mat(ndim, axis1, axis2, theta):
    if axis1 >= ndim:
        raise ValueError("axis1 is out of range!")
    if axis2 >= ndim:
        raise ValueError("axis2 is out of range!")
    if axis1 == axis2:
        raise ValueError("invalid plane axis1 and axis2 are the same!")

    mat = np.eye(ndim, dtype=complex)
    mat[axis1, axis1] = np.cos(theta)
    mat[axis2, axis2] = np.cos(theta)
    mat[axis1, axis2] = -np.sin(theta)
    mat[axis2, axis1] = np.sin(theta)
    return mat

# debug
def generate_plane_rot_mat_SYMB(ndim, axis1, axis2, axname = []):
    if axis1 >= ndim:
        raise ValueError("axis1 is out of range!")
    if axis2 >= ndim:
        raise ValueError("axis2 is out of range!")
    if axis1 == axis2:
        raise ValueError("invalid plane axis1 and axis2 are the same!")

    mat = np.eye(ndim)
    mat = np.array(["%.2f" % w for w in mat.reshape(mat.size)], dtype=object).reshape(mat.shape)
    mat[axis1, axis1] = " cos"
    mat[axis2, axis2] = " cos"
    mat[axis1, axis2] = "-sin"
    mat[axis2, axis1] = " sin"
    if len(axname) < ndim:
        axname = list(range(ndim))
    print(f"Rot[{axname[axis1]}, {axname[axis2]}]=")
    print(mat)

def generate_phase_mat(ndim, phases):
    if ndim != len(phases):
        raise ValueError("Invalid number of phase parameters")

    mat = np.eye(ndim, dtype=complex)
    for i in range(ndim):
        mat[i, i] = np.exp(complex(0, phases[i]))
    return mat

# debug
def generate_phase_mat_SYMB(ndim, phases=[]):
    if ndim != len(phases):
        if len(phases) != 0:
            print("Warning: invalid parameters")
        phases = list(range(ndim))

    mat = np.eye(ndim)
    mat = np.array(["%.7f" % w for w in mat.reshape(mat.size)], dtype=object).reshape(mat.shape)
    for i in range(ndim):
        mat[i, i] = f"e^(i{phases[i]})" #np.exp(complex(0, phases[i]))
    print(mat)

def generate_unit_mat(ndim, rots, phases):
    # unitary mats consist of rotations and an optional reflection
    # however, they can be also thought of a rotation and phase tranformation
    # this is in no way an effecient parameratization of unitary matricies (it still requires ndim * ndim parameters)
    # however, I need a differentiable, and easy to implement solution.

    # References:
    # https://en.wikipedia.org/wiki/Unitary_matrix#:~:text=Another%20factorization%20is
    # https://math.stackexchange.com/questions/1402362/rotation-in-4d#:~:text=earthlings%20are%20used%20too.-,Plane%20Of%20Rotation

    # check validity of parameters
    
    if len(phases) != ndim:
        raise ValueError("There should be a phase specified for each dimension")
    
    # the angles are encoded as planes of rotation
    # not the most effecient, however, simple to implement
    if len(rots) != comb(ndim, 2) * 2:
        raise ValueError("Invalid number of angles!")
    
    # now construct the matrix
    mat = np.eye(ndim, dtype=complex)

    # first go the rotations!
    rot_index = 0
    for axis1, axis2 in combinations(range(ndim), 2):
        mat = generate_plane_rot_mat(ndim, axis1, axis2, rots[rot_index]) @ mat
        rot_index += 1

    # next the phases
    mat = generate_phase_mat(ndim, phases) @ mat

    # and finally the second set of rotations
    for axis1, axis2 in combinations(range(ndim), 2):
        mat = generate_plane_rot_mat(ndim, axis1, axis2, rots[rot_index]) @ mat
        rot_index += 1
    
    return mat