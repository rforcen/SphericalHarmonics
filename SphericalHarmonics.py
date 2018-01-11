from math import sin, cos, pi
from multiprocessing import Pool, cpu_count

from predefs import random_code


class SphericalHarmonics:

    def __init__(self, code='01222412', resol=64):
        self.code, self.resol = code, resol
        self.vcode = self.code2vect(self.code)
        self.du, self.dv, self.dx = pi * 2 / resol, pi / resol, 1 / resol

    def code2vect(self, s):  # convert predef code in '8c' format to a int list
        return list(map(int, s))

    def _calcCoord(self, theta, phi):
        r = 0
        r += sin(self.vcode[0] * phi) ** self.vcode[1]
        r += cos(self.vcode[2] * phi) ** self.vcode[3]
        r += sin(self.vcode[4] * theta) ** self.vcode[5]
        r += cos(self.vcode[6] * theta) ** self.vcode[7]

        return r * sin(phi) * cos(theta), r * cos(phi), r * sin(phi) * sin(theta)

    def calcCoord(self, vv):  # generate quad of(x,y,z) for a given point in 'vv'
        i, j = vv // self.resol, vv % self.resol  # linear value to i,j coord
        u, v = i * self.du, j * self.dv

        res = [self._calcCoord(u, v)]

        res += [self._calcCoord(u + self.du, v)]
        res += [self._calcCoord(u + self.du, v + self.dv)]
        res += [self._calcCoord(u, v + self.dv)]

        return res

    def genCoords(self):  # single thread mode
        return list(map(self.calcCoord, range(self.resol * self.resol)))

    def genCoordsMT(self):  # mt mode doubles performance
        with Pool(cpu_count()) as pool:  # map list
            coords = pool.map(self.calcCoord, range(self.resol * self.resol))
            pool.close()
            pool.join()
        return coords

    def coords(self):
        return self.genCoordsMT()

    def random(self):
        self.code = random_code()
        self.vcode = self.code2vect(self.code)
        return self.coords()
