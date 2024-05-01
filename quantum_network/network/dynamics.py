from .network import QuantumSystem
from quantum.core import DensityMatrix, sigmaX, sigmaY, sigmaZ, sigmaMinus, sigmaPlus
from network.runge_kutta import rungeKutta


class DynamicsFunc:
    def updateOperators(self):
        pass


class AnalyticDynamicsFunc(DynamicsFunc):
    def __init__(self):
        pass

class DerivativeDynamicsFunc(DynamicsFunc):
    def calcDerivative(self, init_state: DensityMatrix):
        pass


class EnergyExchangeDynamics(DerivativeDynamicsFunc):
    def __init__(self, system1: QuantumSystem, system2: QuantumSystem, coupling_strength: float):
        super().__init__()
        self.system1 = system1
        self.system2 = system2
        self.coupling_strength = coupling_strength
        self.updateOperators()
        
    def updateOperators(self):
        self.sigma_x_12 = self.system1.generateParentOperator(sigmaX) * self.system2.generateParentOperator(sigmaX)
        self.sigma_y_12 = self.system1.generateParentOperator(sigmaY) * self.system2.generateParentOperator(sigmaY)
        self.hamiltonian = self.coupling_strength * (self.sigma_x_12 + self.sigma_y_12)

    def calcDerivative(self, init_state: DensityMatrix) -> DensityMatrix:
        ro_dot = -1j * self.hamiltonian.commutator(init_state)
        return ro_dot
    
class DampingDynamics(DerivativeDynamicsFunc):
    def __init__(self, system: QuantumSystem, damping_strength: float):
        self.system = system
        self.damping_strength = damping_strength
        self.updateOperators()

    def updateOperators(self):
        self.sigma_minus = self.system.generateParentOperator(sigmaMinus)
        self.sigma_plus = self.system.generateParentOperator(sigmaPlus)
        
    def calcDerivative(self, init_state: DensityMatrix):
        ro_dot = (self.damping_strength / 2) * (
            2 * self.sigma_minus * init_state * self.sigma_plus
            - init_state * self.sigma_plus * self.sigma_minus
            - self.sigma_plus * self.sigma_minus * init_state
        )
        return ro_dot


class PumpingDynamics(DerivativeDynamicsFunc):
    def __init__(self, system1: QuantumSystem, system2: QuantumSystem, gamma_1: float, gamma_2: float):
        self.system1 = system1
        self.system2 = system2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.updateOperators()

    def updateOperators(self):
        self.sigma_plus_1 = self.system1.generateParentOperator(sigmaPlus)
        self.sigma_plus_2 = self.system2.generateParentOperator(sigmaPlus)
        self.sigma_minus_1 = self.system1.generateParentOperator(sigmaMinus)
        self.sigma_minus_2 = self.system2.generateParentOperator(sigmaMinus)

    def calc(self, init_state: DensityMatrix):
        term1 = - ((self.gamma_1 * self.gamma_2) ** 0.5) * self.sigma_plus_2.commutator(self.sigma_minus_1 * init_state)
        term2 = - ((self.gamma_1 * self.gamma_2) ** 0.5) * (init_state * self.sigma_plus_1).commutator(self.sigma_minus_2)
        ro_dot = term1 + term2
        return ro_dot

class QuantumSystemDynamics:
    def __init__(self, system: QuantumSystem):
        self.system = system
        self.dynamic_funcs = []

    def addDynamics(self, dynamics: DynamicsFunc):
        self.dynamic_funcs.append(dynamics)

    def removeDynamics(self, dynamics):
        self.dynamic_funcs.remove(dynamics)

    def updateDynamics(self):
        if self.dynamic_funcs:
            for func in self.dynamic_funcs: func.updateOperators()
        else:
            raise ValueError('Cannot update dynamics: no dynamic functions defined')

    def evolve(self, time: float):
        pass

class AnalyticSystemDynamics(QuantumSystemDynamics):

    def addDynamics(self, dynamics: DynamicsFunc) -> None:
        if len(self.dynamic_funcs) < 1:
            super().addDynamics(dynamics)
        else:
            raise ValueError('Cannot have multiple analytic dynamic functions')

    def evolve(self, time: float):
        return super().evolve(time)


class DerivativeSystemDynamics(QuantumSystemDynamics):
    def evolve(self, time: float, time_step: float):
        total_time = 0
        while total_time < time:
            self.system.state = rungeKutta(self.computeDerivative, time_step, self.system.state)
            total_time += time_step
    
    def computeDerivative(self, state):
        derivative = self.dynamic_funcs[0].calcDerivative(state)
        for func in self.dynamic_funcs[1:]:
            derivative += func.calcDerivative(state)

        return derivative
