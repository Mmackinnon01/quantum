from quantum.core import DensityMatrix
from quantum_model.dynamics_manager import NumericalDynamicsManager

import numpy as np


class System:

    def __init__(self):
        self.dynamics = []
        self.name = None
        self.dynamics_manager = NumericalDynamicsManager(0.01)

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, ro: DensityMatrix):
        self.__state = ro

    @property
    def configuration(self):
        raise NotImplementedError

    def addDynamics(self, dynamicFunc):
        self.dynamics.append(dynamicFunc)

    def getDynamics(self):
        raise NotImplementedError

    def removeDynamics(self, dynamicFunc):
        return self.dynamics.remove(dynamicFunc)


class SingleSystem(System):

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.nsystems = 1

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, ro: DensityMatrix):
        if ro is not None:
            self.__dim = ro.dim
        self.__state = ro

    @property
    def dim(self):
        return self.__dim

    @property
    def configuration(self):
        return [self.dim]

    def getDynamics(self):
        return self.dynamics

    def updateDyanamics(self):
        for dynamicFunc in self.getDynamics():
            dynamicFunc.updateOperators([0], self.dim)


class CompositeSystem(System):

    def __init__(self):
        super().__init__()
        self.subsystems = []
        self.nsystems = 0
        self.__state = DensityMatrix(matrix=np.array([[1]]))

    def __repr__(self):
        return f"Composite Quantum System: {self.name}"

    @property
    def configuration(self):
        config = []
        for system in self.subsystems:
            config += system.configuration
        return config

    @property
    def dim(self):
        return self.__dim

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, ro):
        for system in self.subsystems:
            system.state = None
        self.__state = ro

    def subsystemIndex(self, target_system: System):
        subsystem_index = 0
        for system in self.subsystems:
            if system == target_system:
                return [subsystem_index + i for i in range(system.nsystems)]
            else:
                if type(system) != SingleSystem:
                    if system.subsystemIndex(target_system) is not None:
                        return [
                            subsystem_index + i
                            for i in system.subsystemIndex(target_system)
                        ]
            subsystem_index += system.nsystems
        return None

    def getSubsystem(self, name: str):
        for system in self.subsystems:
            if name == system.name:
                return system
            else:
                if system.getSubsystem(name) is not None:
                    return system.getSubsystem(name)
        return None

    def addSubsystem(self, system: System):
        if system in self.subsystems:
            raise ValueError("System being added is already in this composite system")
        self.subsystems.append(system)
        self.state = self.state.tensor(system.state)
        self.nsystems += system.nsystems
        self.updateDyanamics()

    def removeSubsystem(self, system: System):
        if self.subsystemIndex(system) is None:
            raise ValueError("System to be removed is not in this composite system")

        system_indices = self.subsystemIndex(system)
        config1 = self.configuration
        config2 = self.configuration
        self.subsystems.remove(system)

        subsystem_state = self.state
        for i in reversed(range(self.nsystems)):
            if i not in system_indices:
                subsystem_state = subsystem_state.partialTrace(i, config1)
                config1.pop(i)

        system.state = subsystem_state
        for index in reversed(system_indices):

            self.state = self.state.partialTrace(index, config2)
            config2.pop(index)

        self.nsystems -= system.nsystems
        self.updateDyanamics()

    def getDynamics(self):
        dynamics = self.dynamics

        for system in self.subsystems:
            dynamics += system.getDynamics()

        return dynamics

    def updateDyanamics(self):
        self.dynamics_manager.dynamic_funcs = []
        for dynamicFunc in self.getDynamics():
            systems = dynamicFunc.systems
            indices = []
            for system in systems:
                indices += self.subsystemIndex(system)
            config, dims = self.generateConfiguration(indices)
            dynamicFunc.updateOperators(config, dims)
            self.dynamics_manager.addDynamics(dynamicFunc)

    def generateConfiguration(self, indices):
        dims = self.configuration
        config = [-1 for i in dims]

        for i, index in enumerate(indices):
            config[index] = i

        return config, dims

    def evolve(self, t):
        self.state = self.dynamics_manager.evolve(self.state, t)

    def getSubsystemState(self, subsystem):
        system_indices = self.subsystemIndex(subsystem)
        subsystem_state = self.state
        config = self.configuration

        for i in reversed(range(self.nsystems)):
            if i not in system_indices:
                subsystem_state = subsystem_state.partialTrace(i, config)
                config.pop(i)

        return subsystem_state

    def measureSubsystem(self, subsystem, measurement):
        return self.getSubsystemState(subsystem).measure(measurement)
