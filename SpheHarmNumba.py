import numba as nb
import cppcolormap as cmap
import numpy as np
from math import sin, cos, pi, sqrt

SHDefs = [
    ('resol', nb.int32),
    ('vcode', nb.float32[:]),
    ('du', nb.float32),
    ('dv', nb.float32),
    ('dx', nb.float32),

    ('_coords', nb.float32[:, :, :]),  # resol^2, quads, coord
    ('_normals', nb.float32[:, :]),
    ('_colors', nb.int32[:, :])
]

_colormapNames = np.array(
    'Accent,Dark2,Paired,Spectral,Pastel1,Pastel2,Set1,Set2,Set3,Blues,Greens,Greys,Oranges,Purples,Reds,BuPu,GnBu,PuBu,PuBuGn,PuRd,RdPu,OrRd,RdOrYl,YlGn,YlGnBu,YlOrRd,BrBG,PuOr,RdBu,RdGy,RdYlBu,RdYlGn,PiYG,PRGn'.split(
        ','))


@nb.jitclass(SHDefs)
class SpheHarmNumba:
    def __init__(self, vcode, resol, colors):
        self.resol = resol
        self.vcode = vcode
        self.du, self.dv, self.dx = pi * 2. / resol, pi / resol, 1. / resol

        self._coords = np.zeros(shape=(resol * resol, 4, 3), dtype=nb.float32)
        self._normals = np.zeros(shape=(resol * resol, 3), dtype=nb.float32)
        self._colors = colors

    def __calcCoord(self, vv):  # generate quad of(x,y,z) for a given point in 'vv'
        def calcPoint(theta, phi):
            r = sin(self.vcode[0] * phi) ** self.vcode[1] + cos(self.vcode[2] * phi) ** self.vcode[3] + \
                sin(self.vcode[4] * theta) ** self.vcode[5] + cos(self.vcode[6] * theta) ** self.vcode[7]

            return r * sin(phi) * cos(theta), r * cos(phi), r * sin(phi) * sin(theta)

        i, j = vv // self.resol, vv % self.resol  # linear value to i,j coord
        u, v = i * self.du, j * self.dv

        return [calcPoint(u, v)
                for u, v in ((u, v), (u + self.du, v), (u + self.du, v + self.dv), (u, v + self.dv))]

    def coords(self):
        for i in range(self.resol * self.resol):
            crd = self.__calcCoord(i)
            for q in range(4):
                self._coords[i][q] = crd[q]
        return self._coords

    def __normQuad(self, quad):  # quad of coords 4x3
        def cross(vec1, vec2):
            """ Calculate the cross product of two 3d vectors. """
            a1, a2, a3 = vec1[0], vec1[1], vec1[2]
            b1, b2, b3 = vec2[0], vec2[1], vec2[2]
            return np.array([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1], dtype=nb.float32)

        def normalize(v):
            return v / (np.linalg.norm(v) + 1e-16)

        return normalize(cross(quad[1] - quad[0], quad[2] - quad[0]))

    def normals(self):  # qcoords are quads of coords
        for i in range(self.resol * self.resol):
            self._normals[i] = self.__normQuad(self._coords[i])
        return self._normals

    def colors(self):
        return self._colors

    def setColors(self, ca):  # resol^2 , 3 int32
        self._colors = ca

    def random_coords(self):
        self.vcode = np.array(list(np.random.randint(0, 8, size=8)), dtype=nb.float32)
        return self.coords()


def str2floatarray(code) -> [nb.float32]:
    return np.array(list(map(nb.float32, code)))


def colorMap(type, resol):
    return cmap.colormap(type, resol * resol)


if __name__ == '__main__':
    resol = 128 * 4

    sh = SpheHarmNumba(vcode=str2floatarray('11224422'), resol=resol, colors=colorMap(type='Accent', resol=resol))
    coords = sh.coords()
    normals = sh.normals()
