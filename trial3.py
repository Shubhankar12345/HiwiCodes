import numpy as np

Arr = np.array([[1,2,3],[4,5,6], [7,8,9]])
tots = Arr.shape[0]*Arr.shape[1]

print(np.random.choice(Arr.flatten(),6, replace = False).reshape(3,2))