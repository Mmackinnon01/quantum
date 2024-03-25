import numpy as np
import cmath as math


def randomUnitary(dimension):
    phase = math.exp(1j*np.random.rand()*2*math.pi)
    rotation_operators = []
    for i in range(dimension-1):
        rotation_operators.append(compositeRotationOperator(i+1, dimension))

    if len(rotation_operators) > 1:
        unitary = phase * np.linalg.multi_dot(rotation_operators)
    else:
        unitary = phase * rotation_operators[0]

    return unitary


def compositeRotationOperator(n, dimension):
    rotations = []
    for i in range(n):
        i += 1
        phi = np.random.rand() * math.pi / 2
        psi = np.random.rand() * math.pi * 2
        if n == i:
            chi = np.random.rand() * math.pi * 2
        else:
            chi = 0
        rotations.append(rotationOperator(n-i, n, dimension, phi, psi, chi))

    if len(rotations) > 1:                      
        compositeRotation = np.linalg.multi_dot(rotations)
    else:
        compositeRotation = rotations[0]

    return compositeRotation


def rotationOperator(i, j, n, phi, psi, chi):
    operator = np.zeros((n, n)).astype(complex)
    for k in range(n):
        for l in range(n):
            if k == l:
                if k == i:
                    operator[k, l] = math.exp(1j*psi) * math.cos(phi)
                elif k == j:
                    operator[k, l] = math.exp(-1j*psi) * math.cos(phi)
                else:
                    operator[k, l] = 1
            elif i == k and j == l:
                operator[k, l] = math.exp(1j*chi)*math.sin(phi)
            elif i == l and j == k:
                operator[k, l] = -math.exp(-1j*chi)*math.sin(phi)
    return operator
