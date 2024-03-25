from quantum.core import Operator

import numpy as np

sigmaI = Operator(np.array([[1,0],[0,1]]))
sigmaX = Operator(np.array([[0,1], [1,0]]))
sigmaY = Operator(np.array([[0, -1j], [1j, 0]]))
sigmaZ = Operator(np.array([[1,0], [0,-1]]))