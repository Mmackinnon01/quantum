from .system import System
from quantum.core import (
    DensityMatrix,
    sigmaX,
    sigmaY,
    sigmaZ,
    sigmaMinus,
    sigmaPlus,
    lambda1,
    lambda2,
    lambda6,
    lambda7,
)


class DynamicsFunc:

    def __init__(self):
        self.operators = []

    def updateOperators(self, config, dims):
        for operator in self.operators:
            operator.transform(config, dims)


class AnalyticDynamicsFunc(DynamicsFunc):
    def __init__(self):
        pass


class DerivativeDynamicsFunc(DynamicsFunc):
    def calcDerivative(self, init_state: DensityMatrix):
        pass


class EnergyExchangeDynamics(DerivativeDynamicsFunc):
    def __init__(self, system1: System, system2: System, coupling_strength: float):
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.systems = [self.system1, self.system2]
        self.coupling_strength = coupling_strength
        self.hamiltonian = self.coupling_strength * (
            sigmaX.tensor(sigmaX) + sigmaY.tensor(sigmaY)
        )
        self.operators = [self.hamiltonian]

    def calcDerivative(self, init_state: DensityMatrix) -> DensityMatrix:
        ro_dot = -1j * self.hamiltonian.commutator(init_state)
        return ro_dot


class QutritQubitExchangeDynamics(DerivativeDynamicsFunc):
    def __init__(self, system1: System, system2: System, coupling_strength: float):
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.systems = [self.system1, self.system2]
        self.coupling_strength = coupling_strength
        self.hamiltonian = self.coupling_strength * (
            (lambda1 + lambda6).tensor(sigmaX) + (lambda2 + lambda7).tensor(sigmaY)
        )
        self.operators = [self.hamiltonian]

    def calcDerivative(self, init_state: DensityMatrix) -> DensityMatrix:
        ro_dot = -1j * self.hamiltonian.commutator(init_state)
        return ro_dot
