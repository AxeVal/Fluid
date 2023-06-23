import numpy as np

from .Calculations import diffuse, advect, project
from src.Properties import *


class FluidCube:
    def __init__(self, dt, diffusion, viscosity, NN = N):
        """
        Инициализирует объект FluidCube.

        Аргументы:
            dt: Шаг времени.
            diffusion: Коэффициент диффузии (скорость жидкости в среде).
            viscosity: Вязкость.

        Примечания:
            - Создает объект FluidCube с заданными параметрами и инициализирует
              все необходимые массивы для моделирования жидкости.
        """
        self.NN = NN
        self.dt   = dt        # шаг времени
        self.diff = diffusion # диффузия, скорость жидкости в среде
        self.visc = viscosity # вязкость

        self.s       = np.zeros((NN, NN))
        self.density = np.zeros((NN, NN)) # плотность

        self.Vx = np.zeros((NN, NN)) # текущая скорость по x
        self.Vy = np.zeros((NN, NN)) # ...              по y

        self.Vx0 = np.zeros((NN, NN)) # предыдущая скорость по x
        self.Vy0 = np.zeros((NN, NN)) # ...                 по y

    def addDensity(self, x: int, y: int, amount: float):
        """
        Добавляет плотность в указанную точку с заданным количеством.

        Аргументы:
            x: Координата по оси X.
            y: Координата по оси Y.
            amount: Количество плотности для добавления.

        Примечания:
            - Изменяет значение плотности в указанной точке на заданное количество.
        """
        x = int(x)
        y = int(y)
        self.density[x, y] += amount

    def addVelocity(self, x, y, amountX, amountY):
        """
        Добавляет скорость в указанную точку с заданным количеством.

        Аргументы:
            x: Координата по оси X.
            y: Координата по оси Y.
            amountX: Количество скорости по оси X для добавления.
            amountY: Количество скорости по оси Y для добавления.

        Примечания:
            - Изменяет значение скорости в указанной точке на заданное количество.
        """
        index = (x, y)
        self.Vx[index] += amountX
        self.Vy[index] += amountY

    def step(self):
        """
        Выполняет шаг моделирования жидкости.

        Примечания:
            - Выполняет последовательность шагов для обновления плотности и скорости жидкости.
        """
        self.Vx0 = diffuse(1, self.Vx0, self.Vx, self.visc, self.dt, self.NN)
        self.Vy0 = diffuse(2, self.Vy0, self.Vy, self.visc, self.dt, self.NN)

        self.Vx0, self.Vy0, self.Vx, self.Vy = project(self.Vx0, self.Vy0, self.Vx, self.Vy, self.NN)

        self.Vx = np.clip(advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt, self.NN), -5, 1)
        self.Vy = np.clip(advect(2, self.Vy, self.Vy0, self.Vy0, self.Vx0, self.dt, self.NN), -5, 1)

        self.Vx, self.Vy, self.Vx0, self.Vy0 = project(self.Vx, self.Vy, self.Vx0, self.Vy0, self.NN)

        self.s = diffuse(0, self.s, self.density, self.diff, self.dt, self.NN)
        self.density = advect(0, self.density, self.s, self.Vx, self.Vy, self.dt, self.NN)
