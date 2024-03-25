import matplotlib.pyplot as plt
from quantum.core import DensityMatrix
import math
import numpy as np

class DensityMatrixVisualiser:

    def __init__(self):
        self.matrices = []

    def addMatrix(self, matrix: DensityMatrix):
        self.matrices.append(matrix)

    def plot(self, title = ""):

        self.figure = plt.figure(figsize=(12, 8))

        for i, matrix in enumerate(self.matrices):
            plt.figtext(0.5, 0.93 - (0.9/len(self.matrices)) * i, matrix.name, ha='center', va='center', fontsize=14)
            plt.subplot(len(self.matrices), 2, 2*i + 1)
            plt.imshow(np.real(matrix.matrix), cmap='Reds', interpolation='none')
            plt.colorbar()
            plt.title("Real Components")
            plt.subplot(len(self.matrices), 2, (2*i)+2)
            plt.imshow(np.imag(matrix.matrix), cmap='Reds', interpolation='none')
            plt.colorbar()
            plt.title("Complex Components")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

