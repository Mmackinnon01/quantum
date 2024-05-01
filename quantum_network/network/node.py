from quantum import DensityMatrix


class Node:
    def __init__(self, dim: int, init_state: DensityMatrix):
        self.dim = dim
        self.init_state = init_state

    