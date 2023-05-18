import numpy as np
from math import floor
from numba import njit
import pygame as pg


N = 100
iter = 10
SCALE = 8

def IX(x: int, y: int):
    return int(x + y * N)

@njit
def set_bnd(b, x):
    x[0, :]   = -x[1, :]  if b == 1 else x[1, :]
    x[-1, :]  = -x[-2, :] if b == 1 else x[-2, :]
    x[:, 0]   = -x[:, 1]  if b == 2 else x[:, 1]
    x[:, -1]  = -x[:, -2] if b == 2 else x[:, -2]

    x[0, 0]   = (x[1, 0]   + x[0, 1])   / 2
    x[0, -1]  = (x[1, -1]  + x[0, -2])  / 2
    x[-1, 0]  = (x[-2, 0]  + x[-1, 1])  / 2
    x[-1, -1] = (x[-2, -1] + x[-1, -2]) / 2
    return x

def advect(b, d, d0, Vx, Vy, dt):
    dtx = dt * (N - 2)
    dty = dt * (N - 2)

    tmp1 = dtx * Vx
    tmp2 = dty * Vy
    i = np.arange(0, N)
    j = np.arange(0, N)
    j, i = np.meshgrid(i, j)

    x = i - tmp1
    y = j - tmp2
    x[x < .5]     = 0.5
    x[x > N + .5] = N + 0.5
    y[y < .5]     = 0.5
    y[y > N + .5] = N + 0.5
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
    d = set_bnd(b, d)
    return d

@njit
def lin_solve(b, x, x0, a, c):
    cRecip = 1.0 / c
    for k in range(0, iter, 1):
        for j in range(1, N-1, 1):
            for i in range(1, N-1, 1):
                x[i, j] = (x0[i, j] + a * (x[i+1, j] +
                                           x[i-1, j] +
                                           x[i, j+1] +
                                           x[i, j-1])) * cRecip
        x = set_bnd(b, x)
    return x

def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N-2) * (N-2)
    x = lin_solve(b, x, x0, a, 1+6*a)
    return x

@njit
def project(velocX, velocY, p, div):
    for j in range(1, N-1):
        for i in range(1, N-1):
            div[i, j] = -0.5 * (velocX[i+1, j] -
                                    velocX[i-1, j] +
                                    velocY[i, j+1] -
                                    velocY[i, j-1]) / N
            p[i, j] = 0
    div = set_bnd(0, div)
    p = set_bnd(0, p)
    p = lin_solve(0, p, div, 1, 6)

    for j in range(1, N-1):
        for i in range(1, N-1):
            velocX[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j] +
                                   p[i, j+1] - p[i, j-1]) * N
            velocY[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j] +
                                   p[i, j+1] - p[i, j-1]) * N
    velocX = set_bnd(1, velocX)
    velocY = set_bnd(2, velocY)
    return velocX, velocY, p, div

class FluidCube:
    def __init__(self, dt, diffusion, viscosity):
        self.dt   = dt        # шаг времени
        self.diff = diffusion # диффузия, скорость жидкости в среде
        self.visc = viscosity # вязкость

        self.s       = np.zeros((N, N))
        self.density = np.zeros((N, N)) # плотность

        self.Vx = np.zeros((N, N)) # текущая скорость по x
        self.Vy = np.zeros((N, N)) # ...              по y

        self.Vx0 = np.zeros((N, N)) # предыдущая скорость по x
        self.Vy0 = np.zeros((N, N)) # ...                 по y

    def addDensity(self, x: int, y: int, amount: float):
        x = int(x)
        y = int(y)
        self.density[x, y] += amount

    def addVelocity(self, x, y, amountX, amountY):
        index = (x, y)
        self.Vx[index] += amountX
        self.Vy[index] += amountY

    def step(self):
        self.Vx0 = diffuse(1, self.Vx0, self.Vx, self.visc, self.dt)
        self.Vy0 = diffuse(2, self.Vy0, self.Vy, self.visc, self.dt)

        self.Vx0, self.Vy0, self.Vx, self.Vy = project(self.Vx0, self.Vy0, self.Vx, self.Vy)

        self.Vx = advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt)
        self.Vy = advect(2, self.Vy, self.Vy0, self.Vy0, self.Vx0, self.dt)

        self.Vx, self.Vy, self.Vx0, self.Vy0 = project(self.Vx, self.Vy, self.Vx0, self.Vy0)

        self.s = diffuse(0, self.s, self.density, self.diff, self.dt)
        self.density = advect(0, self.density, self.s, self.Vx, self.Vy, self.dt)
