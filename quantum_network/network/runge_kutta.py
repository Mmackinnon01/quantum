def rungeKutta(f, h: float, state):
    k1 = runge_kutta_1(f, h, state)
    k2 = runge_kutta_2(f, h, state, k1)
    k3 = runge_kutta_2(f, h, state, k2)
    k4 = runge_kutta_3(f, h, state, k3)
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def runge_kutta_1(f, h: float, state):
    return f(state)

def runge_kutta_2(f, h: float, state, kn_minus_1):
    return f(state + h * (kn_minus_1 / 2))

def runge_kutta_3(f, h: float, state, k3):
    return f(state + h * k3)