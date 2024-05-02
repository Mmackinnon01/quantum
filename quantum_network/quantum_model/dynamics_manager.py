from quantum.core import DensityMatrix
from quantum_model.runge_kutta import rungeKutta


class DynamicsManager:
    def __init__(self, timestep):
        self.timestep = timestep
        self.dynamic_funcs = []

    def addDynamics(self, dynamics):
        self.dynamic_funcs.append(dynamics)

    def removeDynamics(self, dynamics):
        self.dynamic_funcs.remove(dynamics)

    def evolve(self, time: float):
        pass


class NumericalDynamicsManager(DynamicsManager):
    def evolve(self, state: DensityMatrix, time: float):
        total_time = 0
        while total_time < time:
            state = rungeKutta(self.computeDerivative, self.timestep, state)
            total_time += self.timestep
        return state

    def computeDerivative(self, state):
        derivative = 0
        for func in self.dynamic_funcs:
            derivative += func.calcDerivative(state)

        return derivative
