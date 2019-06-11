import cppcolormap as cmap
import numpy as np
from math import sin, cos, pi, sqrt
from multiprocessing import Pool, cpu_count
from random import randint

from predefs import random_code

_colormapNames = np.array(
    'Accent,Dark2,Paired,Spectral,Pastel1,Pastel2,Set1,Set2,Set3,Blues,Greens,Greys,Oranges,Purples,Reds,BuPu,GnBu,PuBu,PuBuGn,PuRd,RdPu,OrRd,RdOrYl,YlGn,YlGnBu,YlOrRd,BrBG,PuOr,RdBu,RdGy,RdYlBu,RdYlGn,PiYG,PRGn'.split(
        ','))


class SphericalHarmonics:

    def __init__(self, code='01222412', resol=128):
        self.code = code
        self.resol = resol
        self.vcode = self.code2vect(code)
        self.du, self.dv, self.dx = pi * 2 / resol, pi / resol, 1 / resol
        self.colormapNames = _colormapNames
        self._coords = None
        self._normals = None
        self._colors = None

    def code2vect(self, s) -> int:  # convert predef code in '8c' format to a int list
        return list(map(int, s))

    def calcCoord(self, vv):  # generate quad of(x,y,z) for a given point in 'vv'
        def _calcCoord(theta, phi):
            r = sum([sin(self.vcode[0] * phi) ** self.vcode[1],
                     cos(self.vcode[2] * phi) ** self.vcode[3],
                     sin(self.vcode[4] * theta) ** self.vcode[5],
                     cos(self.vcode[6] * theta) ** self.vcode[7]])

            return r * sin(phi) * cos(theta), r * cos(phi), r * sin(phi) * sin(theta)

        i, j = vv // self.resol, vv % self.resol  # linear value to i,j coord
        u, v = i * self.du, j * self.dv

        return [_calcCoord(u, v)
                for u, v in ((u, v), (u + self.du, v), (u + self.du, v + self.dv), (u, v + self.dv))]

    def coords(self):
        if cpu_count() > 3:  # multi thread mode
            with Pool(cpu_count()) as pool:
                self._coords = pool.map(self.calcCoord, range(self.resol * self.resol))
                pool.close()
                pool.join()
        else:  # st mode
            self._coords = list(map(self.calcCoord, range(self.resol * self.resol)))
        return self._coords

    def normQuad(self, quad):
        def normVect(v):
            len = sqrt(sum([c * c for c in v]))
            if len != 0:
                return [c / len for c in v]
            else:
                return [0., 0., 0.]

        pa = [x - y for x, y in zip(quad[1], quad[0])]
        pb = [x - y for x, y in zip(quad[2], quad[0])]
        n = [pa[i] * pb[j] - pa[j] * pb[i] for i, j in ((1, 2), (2, 0), (0, 1))]

        return normVect(n)

    def normals(self):  # qcoords are quads of coords
        if cpu_count() > 3:
            with Pool(cpu_count()) as pool:  # map list
                self._normals = pool.map(self.normQuad, self._coords)
                pool.close()
                pool.join()
        else:
            self._normals = list(map(self.normQuad, self._coords))
        return self._normals

    def colors(self, cm=None):
        try:
            _cm = cm
            if cm is None:  # random
                _cm = self.colormapNames[randint(0, len(self.colormapNames))]
            else:
                if type(cm) == int:  # by index
                    _cm = self.colormapNames[cm]
            self._colors = cmap.colormap(_cm, self.resol ** 2)
        except:
            pass
        return self._colors

    def randomCode(self):  # select a random preset code
        self.code = random_code()
        self.vcode = self.code2vect(self.code)
        return self.coords()

    def random(self):  # generate a code w/random set
        self.code = ''.join([str(randint(0, 8)) for i in range(8)])
        self.vcode = self.code2vect(self.code)
        return self.coords()
