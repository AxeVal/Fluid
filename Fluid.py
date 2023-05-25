import numpy as np
from math import floor
from numba import njit


# гиперпарамеитры
N = 80         # размер рабочей матрицы
iterations = 4 # инетерации
SCALE = 10     # масштабирование окна
bounds = False # наличие отражения от стенок


# set bounds
@njit
def set_bnd(b, x):
    x[0,  :] = (-x[1,  :] if b == 1 else x[1,  :]) * bounds
    x[-1, :] = (-x[-2, :] if b == 1 else x[-2, :]) * bounds
    x[:,  0] = (-x[:,  1] if b == 2 else x[:,  1]) * bounds
    x[:, -1] = (-x[:, -2] if b == 2 else x[:, -2]) * bounds

    x[0,   0] = (x[1,   0] + x[0,   1]) * 0.5 * bounds
    x[0,  -1] = (x[1,  -1] + x[0,  -2]) * 0.5 * bounds
    x[-1,  0] = (x[-2,  0] + x[-1,  1]) * 0.5 * bounds
    x[-1, -1] = (x[-2, -1] + x[-1, -2]) * 0.5 * bounds
    return x

def advect(b, d, d0, Vx, Vy, dt):
    i = np.arange(0, N)
    j = np.arange(0, N)
    j, i = np.meshgrid(i, j)

    x = i - dt * (N - 2) * Vx
    y = j - dt * (N - 2) * Vy
    x[x < 0.5]     = 0.5
    x[x > N + 0.5] = N + 0.5
    y[y < 0.5]     = 0.5
    y[y > N + 0.5] = N + 0.5
    i0 = np.floor(x).astype(np.int32)
    i1 = i0 + 1
    j0 = np.floor(y).astype(np.int32)
    j1 = j0 + 1
    i0 = np.clip(i0, 0, N - 1)
    i1 = np.clip(i1, 0, N - 1)
    j0 = np.clip(j0, 0, N - 1)
    j1 = np.clip(j1, 0, N - 1)

    s1 = x - i0
    s0 = 1 - s1
    t1 = y - j0
    t0 = 1 - t1
    d = (s0 * (t0 * d0[i0, j0] +
               t1 * d0[i0, j1]) +
         s1 * (t0 * d0[i1, j0] +
               t1 * d0[i1, j1]))
    return set_bnd(b, d)

@njit
def lin_solve(b, x, x0, a, c):
    cRecip = 1.0 / c
    for iter in range(0, iterations, 1):
        for i in range(1, N-1, 1):
            for j in range(1, N-1, 1):
                x[i, j] = (x0[i, j] + a * (x[i+1, j] + x[i-1, j] +
                                           x[i, j+1] + x[i, j-1])) * cRecip
        x = set_bnd(b, x)
    return x

def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N-2) * (N-2)
    x = lin_solve(b, x, x0, a, 1+6*a)
    return x

@njit
def project(Vx, Vy, p, div):
    for j in range(1, N-1, 1):
        for i in range(1, N-1, 1):
            div[i, j] = -0.5 * (Vx[i+1, j] - Vx[i-1, j] +
                                Vy[i, j+1] - Vy[i, j-1]) / N
            p[i, j] = 0
    div = set_bnd(0, div)
    p = set_bnd(0, p)
    p = lin_solve(0, p, div, 1, 6)

    for j in range(1, N-1):
        for i in range(1, N-1):
            Vx[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j] +
                               p[i, j+1] - p[i, j-1]) * N
            Vy[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j] +
                               p[i, j+1] - p[i, j-1]) * N
    Vx = set_bnd(1, Vx)
    Vy = set_bnd(2, Vy)
    return Vx, Vy, p, div


class FluidCube:
    def __init__(self, dt, diffusion, viscosity):
        self.dt   = dt        # шаг времени
        self.diff = diffusion # диффузия, скорость жидкости в среде
        self.visc = viscosity # вязкость

        self.s       = np.zeros(shape=(N, N))
        self.density = np.zeros(shape=(N, N)) # плотность

        self.Vx = np.zeros(shape=(N, N)) # текущая скорость по x
        self.Vy = np.zeros(shape=(N, N)) # ...              по y

        self.Vx0 = np.zeros(shape=(N, N)) # предыдущая скорость по x
        self.Vy0 = np.zeros(shape=(N, N)) # ...                 по y

    def addDensity(self, x, y, amount):
        x = int(x) % N
        y = int(y) % N
        self.density[x, y] += amount

    def addVelocity(self, x, y, amountX, amountY):
        self.Vx[int(x), int(y)] += amountX
        self.Vy[int(x), int(y)] += amountY

    def step(self):
        self.Vx0 = diffuse(1, self.Vx0, self.Vx, self.visc, self.dt)
        self.Vy0 = diffuse(2, self.Vy0, self.Vy, self.visc, self.dt)

        self.Vx0, self.Vy0, self.Vx, self.Vy = project(self.Vx0, self.Vy0, self.Vx, self.Vy)

        self.Vx = advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt)
        self.Vy = advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0, self.dt)

        self.Vx, self.Vy, self.Vx0, self.Vy0 = project(self.Vx, self.Vy, self.Vx0, self.Vy0)

        self.s       = diffuse(0, self.s, self.density, self.diff, self.dt)
        self.density = advect(0, self.density, self.s, self.Vx, self.Vy, self.dt)
