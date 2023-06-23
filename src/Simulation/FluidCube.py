import numpy as np

from .Calculations import diffuse, advect, project
from Properties import *

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

        self.Vx = np.clip(advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt), -5, 1)
        self.Vy = np.clip(advect(2, self.Vy, self.Vy0, self.Vy0, self.Vx0, self.dt), -5, 1)

        self.Vx, self.Vy, self.Vx0, self.Vy0 = project(self.Vx, self.Vy, self.Vx0, self.Vy0)

        self.s = diffuse(0, self.s, self.density, self.diff, self.dt)
        self.density = advect(0, self.density, self.s, self.Vx, self.Vy, self.dt)
