import numpy as np

def softmax(x):
    if len(x.shape) == 1:
        return np.exp(x) / np.sum(np.exp(x))
    elif len(x.shape) == 2:
        return np.array([softmax(x_i) for x_i in x])
    else:
        raise ValueError("Invalid input shape!")

def states_p(input_vecs, expected_vecs, epsilon=1E-8):
    dists = np.zeros((input_vecs.shape[0], expected_vecs.shape[0]))

    for i in range(expected_vecs.shape[0]):
        dists[:, i] = np.sqrt(np.sum((input_vecs - np.tile(expected_vecs[i], (input_vecs.shape[0], 1)))**2, axis=1))
   
    dists += np.ones(dists.shape) * epsilon # add epsilon so that the inverse won't explode!
    
    return softmax(dists ** -1)

# measured = np.array([[0,0,1.],[1.,0,0],[0,1.,0]])
# targets = np.array([[0,0,1.],[1.,0,0]])

# outp = states_p(measured, targets, .01)
# print(outp)