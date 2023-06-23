import numpy as np
from numba import njit

from Properties import *

def IX(x: int, y: int):
    return int(x + y * N)

@njit
def set_bnd(b, x):
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

@njit(parallel=True)
def lin_solve(b, x, x0, a, c):
    cRecip = 1.0 / c
    for k in range(iter):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] +
                                               x[2:, 1:-1] +
                                               x[1:-1, :-2] +
                                               x[1:-1, 2:])) * cRecip
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