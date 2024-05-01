from quantum.core import DensityMatrix, Operator

import numpy as np
from typing import TypeVar
T = TypeVar('T', bound='QuantumSystem')


class QuantumSystem:

    def __init__(self, n_qubits = 0):
        self.subsystems = []
        self.parent = None
        self.state = None
        self.name = None
        self.n_qubits = n_qubits

    @property
    def state(self) -> DensityMatrix:
        if self.__state is None:
            if self.parent:
                self.__setState(self.parent.subsystemState(self))
        return self.__state
    
    @state.setter
    def state(self, state: DensityMatrix):
        if self.parent and state is not None:
            raise ValueError('Cannot set state of system that has a parent')
        elif self.subsystems:
            for system in self.subsystems:
                system.state = None 
        self.__state = state

    # State setting method for use only within this class
    def __setState(self, state: DensityMatrix):
        if self.subsystems:
            for system in self.subsystems:
                system.state = None 
        self.__state = state

    def addSubsystem(self, subsystem: 'QuantumSystem'):
        if subsystem.parent:
            return ValueError("This subsystem already has a parent")
        else:
            if subsystem.state is not None:
                self.state = self.state.tensor(subsystem.state)
            subsystem.parent = self
            self.subsystems.append(subsystem)
            self.n_qubits += subsystem.n_qubits

    def removeSubsystem(self, system: 'QuantumSystem'):
        total_subsystems = len(self.subsystems)
        for subsystem in self.subsystems:
            if subsystem == system:
                system.parent = None
                system.state = self.traceSubsystem(system)
                self.subsystems.remove(system)
                self.n_qubits -= system.n_qubits
        
        if len(self.subsystems) == total_subsystems:
            return ValueError("The system you are trying to remove does not exist")
        
    def traceSubsystem(self, system: 'QuantumSystem'):
        qubits = [subsystem.n_qubits for subsystem in self.subsystems]
        system_index = [i for i, subsystem in enumerate(self.subsystems) if subsystem == system][0]
        
        state = self.state

        for i in reversed(range(sum(qubits[:system_index]), sum(qubits[:system_index+1]))):
            state = state.partialTrace(i)

        state.system_dimensions = [self.system_dimensions[system_index]]
        return state
        
    def subsystemState(self, subsystem: 'QuantumSystem'):
        if self.state is None and self.parent:
            self.state = self.parent.subsystemState(self)
        other_system_indices = [i for i, system in enumerate(self.subsystems) if system != subsystem]

        state = self.state
        print(state)
        for system_index in reversed(other_system_indices):
            print(system_index)
            state = state.partialTrace(system_index)

        return state

    def generateSubsystemOperator(self, subsystem: 'QuantumSystem', operator: Operator) -> Operator:
        if subsystem == self:
            return operator
        elif not self.subsystems:
            return Operator(np.eye(N=self.state.matrix.shape[0], M=self.state.matrix.shape[1]))
        else:
            op = self.subsystems[0].generateSubsystemOperator(subsystem, operator)
            for system in self.subsystems[1:]:
                op = op.tensor(system.generateSubsystemOperator(subsystem, operator))
            return op

    def generateParentOperator(self, operator: Operator, subsystem=None) -> Operator:
        if subsystem is None:
            subsystem = self
        if self.parent:
            return self.parent.generateParentOperator(operator, subsystem)
        else:
            return self.generateSubsystemOperator(subsystem, operator)

    def measure(self, operator, subsystem=None):
        if self.parent:
            return self.parent.measure(operator, self)
        elif subsystem is None:
            return self.state.measure(operator)
        else:
            op = self.subsystems[0].generateSubsystemOperator(subsystem, operator)
            for system in self.subsystems[1:]:
                op = op.tensor(system.generateSubsystemOperator(subsystem, operator))
            return self.state.measure(op)

