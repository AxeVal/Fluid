from Fluid import *
import pygame as pg
import sys
import cv2


WIDTH = HEIGHT = N
FPS = 50

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def start():
    pg.init()
    pg.mixer.init()
    sc = pg.display.set_mode((WIDTH*SCALE, HEIGHT*SCALE))
    pg.display.set_caption("Fluid simulation")
    clock = pg.time.Clock()

    fluid = FluidCube(1e-2, 1e-9, 1e-11)

    sc.fill(BLACK)
    pg.display.update()

    surf = pg.surfarray.make_surface(cv2.resize(fluid.density,
                                                (WIDTH*SCALE,
                                                 HEIGHT*SCALE)))

    flag = False
    while True:
        for i in pg.event.get():
            if i.type == pg.KEYDOWN and i.key == pg.K_ESCAPE:
                sys.exit()
            elif i.type == pg.MOUSEBUTTONDOWN:
                flag = True
                pos0 = pg.mouse.get_pos()
            elif i.type == pg.MOUSEBUTTONUP:
                flag = False

        if flag:
            pos = pg.mouse.get_pos()
            fluid.addDensity(pos0[0]//SCALE, pos0[1]//SCALE, 4000)
            fluid.addVelocity(pos0[0]//SCALE, pos0[1]//SCALE,
                              (pos[0]-pos0[0])//SCALE, (pos[1]-pos0[1])//SCALE)

        fluid.step()
        pg.surfarray.blit_array(surf, cv2.resize(fluid.density, (WIDTH*SCALE,
                                                                 HEIGHT*SCALE)))
        sc.blit(surf, (0, 0))
        pg.display.update()


def main():
    start()

if __name__ == '__main__':
    main()
