import numpy as np
import scipy
import math
import itertools

from quantum.unitary import randomUnitary


class Matrix:
    
    def __init__(self, matrix = None, name=None):
        self.matrix = matrix
        self.name = name

    def __repr__(self):
        return f'{self.matrix.round(5)}'

    def __add__(self, val):
        if isinstance(val, int):
            return self
        return self.returnNew(self.matrix + val.matrix)
    
    def __radd__(self, val):
        if isinstance(val, int):
            return self
        return self.returnNew(self.matrix + val.matrix)
    
    def __rmul__(self, val):
        return self.returnNew(self.matrix * val)
    
    def __sub__(self, val):
        return self.returnNew(self.matrix - val.matrix)
    
    def __mul__(self, val):
        return self.returnNew(self.matrix * val)
    
    def __truediv__(self, val):
        return self.returnNew(self.matrix / val)

    @property
    def matrix(self):
        return self.__matrix
    
    @matrix.setter
    def matrix(self, matrix: np.array):
        self.__matrix = matrix

    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name: str):
        self.__name = name

    def trace(self):
        return np.trace(self.matrix)

    def copy(self):
        return self.returnNew(self.matrix)

    def returnNew(self, matrix):
        raise NotImplementedError()

    def hermConj(self):
        return self.returnNew(np.conjugate(np.transpose(self.matrix)))
    
    def eigenvalues(self):
        return np.linalg.eig(self.matrix)

    def tensor(self, matrices):
        mat = self.matrix
        
        if type(matrices) != list:
            matrices = [matrices]

        for matrix in matrices:
            mat = np.kron(mat, matrix.matrix)

        return self.returnNew(mat)

    def dims(self):
        return self.matrix.shape
    
    def commutator(self, mat):
        return self * mat - mat * self

    def anticommutator(self, op):
        return self * mat + mat * self


class Operator(Matrix):

    def __mul__(self, val):
        if type(val) == DensityMatrix:
            return DensityMatrix(np.matmul(self.matrix, val.matrix))
        elif type(val) == Operator:
            return Operator(np.matmul(self.matrix, val.matrix))
        else:
            return Operator(self.matrix*val)

    def returnNew(self, matrix):
        return Operator(matrix)
    

class DensityMatrix(Matrix):

    def __init__(self, matrix = None, name=None):
        self.n_systems = None
        super().__init__(matrix, name)


    def __mul__(self, val):
        if type(val) == Operator or type(val) == DensityMatrix:
            return DensityMatrix(np.matmul(self.matrix, val.matrix))
        else:
            return DensityMatrix(self.matrix * val)

    @property
    def matrix(self):
        return self.__matrix
    
    @matrix.setter
    def matrix(self, matrix: np.array):
        self.__matrix = matrix
        if self.n_systems is None:
            self.n_systems = int(np.log2(matrix.shape[0]))

    def returnNew(self, matrix):
        return DensityMatrix(matrix)

    def isLegitamate(self) -> bool:
        is_normalised = (round(self.trace(),5) == 1)
        is_semi_positive = (min(self.eigenvalues().eigenvalues) >= -0.00001)
        is_hermitian = (np.round(self.hermConj().matrix,5) == np.round(self.matrix,5)).all()
        return is_normalised and is_semi_positive and is_hermitian

    def partialTranspose(self, system):
        ppt_state = np.zeros(self.matrix.shape).astype("complex")
        minor_chunk_size = 1
        num_major_chunks = 1
        num_minor_chunks = 2

        for i in range(self.n_systems - system - 1):
            minor_chunk_size *= 2

        major_chunk_size = minor_chunk_size * num_minor_chunks

        for i in range(system):
            num_major_chunks *= 2        

        for major_i in range(num_major_chunks):
            for major_j in range(num_major_chunks):
                for minor_i in range(num_minor_chunks):
                    for minor_j in range(num_minor_chunks):
                        if minor_i != minor_j:
                            ppt_state[major_i * major_chunk_size + minor_i * minor_chunk_size: major_i * major_chunk_size + (minor_i+1) * minor_chunk_size, major_j * major_chunk_size + minor_j * minor_chunk_size: major_j * major_chunk_size + (minor_j+1) * minor_chunk_size] = self.matrix[major_i * major_chunk_size + minor_j * minor_chunk_size: major_i * major_chunk_size + (minor_j+1) * minor_chunk_size,major_j * major_chunk_size + minor_i * minor_chunk_size: major_j * major_chunk_size + (minor_i+1) * minor_chunk_size]
                        else:
                            ppt_state[major_i * major_chunk_size + minor_i * minor_chunk_size: major_i * major_chunk_size + (minor_i+1) * minor_chunk_size, major_j * major_chunk_size + minor_j * minor_chunk_size: major_j * major_chunk_size + (minor_j+1) * minor_chunk_size] = self.matrix[major_i * major_chunk_size + minor_j * minor_chunk_size: major_i * major_chunk_size + (minor_j+1) * minor_chunk_size,major_j * major_chunk_size + minor_i * minor_chunk_size: major_j * major_chunk_size + (minor_i+1) * minor_chunk_size]

        return DensityMatrix(ppt_state)

    def negativity(self, degree = -1):
        if degree == self.n_systems or degree == -1:
            return self.averageBipartitionNegativity()
        else:
            return self.subsystemNegativity(degree)
        
    def negativityComponents(self):
        negativities = []
        
        partitions = self.computeBipartitions()

        for partition in partitions:
            negativity = self.bipartitionNegativity(partition)
            negativities.append(negativity)

        return negativities

    def computeBipartitions(self):
        partitions = []

        for partition_size in range(int(self.n_systems/2)):
            partial_partitions = list(itertools.combinations(range(self.n_systems), partition_size+1))
            if partition_size + 1 == int(self.n_systems/2) and self.n_systems % 2 == 0:
                partial_partitions_no_duplicates = []

                for partition in partial_partitions:
                    inverse_partition = [val for val in range(self.n_systems) if val not in partition]
                    if tuple(inverse_partition) not in partial_partitions_no_duplicates:
                        partial_partitions_no_duplicates.append(partition)
                
                partitions.extend(partial_partitions_no_duplicates)
            else:
                partitions.extend(partial_partitions)
        return partitions

    def bipartitionNegativity(self, systems):
        state = self
        for system in systems:
            state = state.partialTranspose(system)

        return max([0,np.real(min(np.linalg.eigvals(state.matrix))) * -2])

    def averageBipartitionNegativity(self):
        negativities = self.negativityComponents()

        return math.prod(negativities)**(1/len(negativities))
    
    def subsystemNegativity(self, degree):
        negativities = []
        for system in range(self.n_systems):
            if self.bipartitionNegativity([system]) > 10e-7:
                negativities.append(None)
            else:
                negativities.append(self.partialTrace(system).negativity(degree))

        return negativities

    def partialTrace(self, trace_system):
        trace_basis = self.generateSubsystemBasis(trace_system)
        sub_state = sum(vector.hermConj() * self * vector for vector in trace_basis)
        return sub_state

    def generateSubsystemBasis(self, trace_system):
        basis = []

        for i in range(2):
            basis_vector = np.zeros(2)
            basis_vector[i] = 1
            basis_vector = Operator(basis_vector)
            if trace_system > 0:
                basis_vector = Operator(np.eye(2**trace_system)).tensor(basis_vector)
            if trace_system < self.n_systems - 1:
                basis_vector = basis_vector.tensor(Operator(np.eye(2**(self.n_systems - trace_system - 1))))

            basis.append(basis_vector.hermConj())

        return basis

    def reduceToSubsystem(self, target_system):
        if target_system >= self.n_systems:
            raise ValueError(f'System {target_system} is out of range for a zero indexed {self.n_systems}-qubit system')
        state = self
        for system in reversed(range(self.n_systems)):
            if system != target_system:
                state = state.partialTrace(system)
        return state

    def vonNeumann(self):
        vn = 0
        for val in self.eigenvalues().eigenvalues:
            if val > 0:
                vn -= val * math.log(np.real(val))
        return np.real(vn)

    def measure(self, operator):
        return np.real((self * operator).trace())
    
    def normalise(self):
        return self/self.trace()

    def linearMap(self, operators: list[Operator]):

        if type(operators) != list:
            operators = [operators]

        return sum(operator * self * operator.hermConj() for operator in operators)
    
    def uhlmannFidelity(self, state):
        sqrt_target = DensityMatrix(scipy.linalg.sqrtm(state.matrix))
        product = sqrt_target * self * sqrt_target
        sqrt_product = scipy.linalg.sqrtm(product.matrix)
        fidelity = (np.trace(sqrt_product))**2
        return np.real(fidelity)
    
    def buresDistance(self, state):
        return 2*(1-self.uhlmannFidelity(state))


class QuantumChannel:

    def __init__(self, sum_operators):
        self.sum_operators = sum_operators
        self.gen = GeneralQubitMatrixGen()
        self.n_qubits = int(math.log2(self.sum_operators[0].matrix.shape[0]))

    def map(self, state: DensityMatrix):
        return sum(operator * state * operator.hermConj() for operator in self.sum_operators)
    
    def unmap(self, state: DensityMatrix):
        return sum(operator.hermConj() * state * operator for operator in self.sum_operators)

    def channelCapacity(self):
        coherent_information = []
        
        for i in range(1000):
            state = self.gen.generateState(n_qubits = self.n_qubits)
            channeled_state = self.map(state)
            coherent_information.append(channeled_state.vonNeumann() - state.vonNeumann())

        return max(coherent_information)
    
    def maximumVonNeumannError(self, target_channel, nstates=1000):
        return max(self.vonNeumannError(self.gen.generateState(n_qubits= self.n_qubits), target_channel) for i in range(nstates))

    def vonNeumannError(self, state, target_channel):
        target_state = target_channel.map(state)
        approx_state = self.map(state)

        target_entropy = target_state.vonNeumann()
        approx_entropy = approx_state.vonNeumann()

        return np.real(np.abs(target_entropy - approx_entropy))/2**(self.n_qubits-1)
    
    def maximumPositivityError(self, target_channel, nstates=1000):
        return max(self.positivityError(self.gen.generateState(n_qubits=self.n_qubits), target_channel) for i in range(nstates))

    def positivityError(self, state, target_channel):
        target_state = target_channel.map(state)
        approx_state = self.map(state)

        target_entanglement = target_state.ppt()
        approx_entanglement = approx_state.ppt()

        return np.real(np.abs(target_entanglement - approx_entanglement))

    def minimumFidelity(self, target_channel, nstates=1000):
        return min(self.fidelity(self.gen.generatePureState(n_qubits= self.n_qubits), target_channel) for i in range(nstates))

    def fidelity(self, state, target_channel):
        mapped = self.map(state)
        unmapped = target_channel.unmap(mapped)
        fidelity = (state * unmapped).trace()
        return np.real(fidelity)
        


class GeneralQubitMatrixGen:

    def __init__(self):
        pass

    def generateState(self, n_qubits = 1):
        separable_state = self.generateSeparableState(n_qubits)
        unitary = DensityMatrix(randomUnitary(2**n_qubits))
        general_state = unitary*separable_state*unitary.hermConj()
        return general_state
    
    def generatePartiallyEntangledState(self, n_qubits, degree):
        if degree > 0:
            entangled_partition = self.generateState(degree)
            if n_qubits == degree:
                return entangled_partition
        if n_qubits > degree:
            separable_partition = self.generateSeparableState(n_qubits - degree)
            if degree == 0:
                return separable_partition
        return entangled_partition.tensor(separable_partition)

    
    def generateMixedState(self, n_qubits=1):
        T = DensityMatrix(np.random.normal(size=(2**n_qubits, 2**n_qubits)) + 1j * \
            np.random.normal(size=(2**n_qubits, 2**n_qubits)))

        density_matrix = (T * T.hermConj()).normalise()

        return density_matrix

    def generateSeparableState(self, n_qubits=4):
        state = 0
        for i in range(n_qubits):
            new_state = self.generateMixedState()
            if type(state) == int:
                state = new_state
            else:
                state = state.tensor(new_state)
        return state
    
    def generatePureState(self, n_qubits=1):
        state = np.random.normal(size=(2**n_qubits,)) + 1j * np.random.normal(size=(2**n_qubits,))
        state = DensityMatrix(np.outer(state, np.conj(state.T)))
        return state.normalise()

    def generateWernerState(self, c=None):
        if c is None:
            c = np.random.rand()
        state = np.eye(4,4) * (1-c)/4 + c * np.array([[0,0,0,0],[0,0.5,0.5,0],[0,0.5,0.5,0],[0,0,0,0]])
        return DensityMatrix(state)
    
    def generatePDState(self, c1, c2, c3):
        state = DensityMatrix(np.eye(4,4)) + c1 * sigmaX.tensor(sigmaX) + c2 * sigmaY.tensor(sigmaY) + c3 * sigmaZ.tensor(sigmaZ)
        return state / 4


class QuantumChannelGenerator:

    def __init__(self):
        pass

    def gen(self, d1 = 2, d2 = 2):
        M = math.ceil(d1/d2)
        ginibre_matrices = self.ginibre(d1, d2, M)#+int(np.random.rand() * 5))
        h_matrix = self.hMatrix(ginibre_matrices)
        kraus = self.krausOperators(ginibre_matrices, h_matrix)
        return QuantumChannel(kraus)
    
    def ginibre(self, d1, d2, num=1):
        return [Operator(np.random.normal(size=(d1,d2)) +1j * np.random.normal(size=(d1,d2))) for i in range(num)]
    
    def hMatrix(self, ginibre_mats):
        return sum([mat.hermConj() * mat for mat in ginibre_mats])
    
    def krausOperators(self, ginibre_mats, H):
        H_inv = Operator(scipy.linalg.sqrtm(np.linalg.inv(H.matrix)))
        return [mat * H_inv for mat in ginibre_mats]
    

sigmaX = Operator(np.array([[0,1], [1,0]]))
sigmaY = Operator(np.array([[0, -1j], [1j, 0]]))
sigmaZ = Operator(np.array([[1,0], [0,-1]]))
sigmaPlus = Operator(np.array([[0, 1], [0, 0]]))
sigmaMinus = Operator(np.array([[0, 0], [1, 0]]))
