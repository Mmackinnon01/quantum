from quantum.core import DensityMatrix

import numpy as np


class System:

    def __init__(self):
        self.dynamics = []
        self.name = None

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


class CompositeSystem(System):

    def __init__(self):
        super().__init__()
        self.subsystems = []
        self.nsystems = 0
        self.__state = DensityMatrix(matrix=np.array([[1]]))

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
                    if system.subsytemIndex(target_system) is not None:
                        return [
                            subsystem_index + i
                            for i in system.subsystemIndex(target_system)
                        ]
            subsystem_index += system.nsystems
        raise ValueError("Target system is not contained in composite system")

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

    def removeSubsystem(self, system: System):
        if self.subsystemIndex(system) is None:
            raise ValueError("System to be removed is not in this composite system")
        system_indices = self.subsystemIndex(system)

        for index in reversed(system_indices):
            system.state = 
            self.state = self.state.partialTrace(index, self.configuration)
            self.configuration.pop(index)
        self.subsystems.remove(system)
        self.nsystems -= system.nsystems

    def getDynamics(self):
        dynamics = self.dynamics

        for system in self.subsystems:
            dynamics += system.getDynamics()

        return dynamics

    def updateDyanamics(self):
        for dynamicFunc in self.getDynamics():
            systems = dynamicFunc.getSystems()
            indices = []
            for system in systems:
                indices += self.subsystemIndex(system)
            config, dims = self.generateConfiguarion(indices)
            dynamicFunc.updateOperators(config, dims)
            system.updateDynamics()

    def generateConfiguration(self, indices):
        dims = self.configuration
        config = [-1 for i in dims]

        for i, index in enumerate(indices):
            config[index] = i

        return config, dims

    def getSubsystemState(self):
        pass
