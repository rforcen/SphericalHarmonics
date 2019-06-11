import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow)

from SphericalHarmonics import SphericalHarmonics
from SpheHarmNumba import SpheHarmNumba, str2floatarray, colorMap
from predefs import random_code
from rendererGL import RendererGL


class SHwidget(RendererGL):
    coords = None
    colors = None
    normals = None
    sh = None
    win = None
    need_compile = True
    gl_compiled_list = 1

    def __init__(self, sh, win):
        super(SHwidget, self).__init__()
        self.sh = sh
        self.win = win
        self.getGeo()

        self.setFocusPolicy(Qt.StrongFocus)  # accepts key events
        self.disp_title()

    def getGeo(self):
        self.coords = self.sh.coords()
        self.colors = self.sh.colors()
        self.normals = self.sh.normals()

    def setCoords(self, coords):
        self.coords = coords
        self.colors = self.sh.colors()
        self.normals = self.sh.normals()
        self.disp_title()
        self.need_compile = True
        self.repaint()

    def disp_title(self):
        self.win.setWindowTitle('Spherical Harmonics, code: ' + ''.join(list(map(str, map(int, self.sh.vcode)))))

    def init(self, gl):
        self.sceneInit(gl)
        gl.glCullFace(gl.GL_FRONT)

    def draw(self, gl):
        def drawLine(gl):
            if self.coords is not None:
                for ic, quad in enumerate(self.coords):
                    gl.glColor3ubv(list(self.colors[ic]))
                    gl.glBegin(gl.GL_LINE_LOOP)
                    for c in quad:
                        gl.glVertex3fv(c)
                    gl.glEnd()

        def drawSolid(gl):
            if self.coords is not None:
                gl.glEnable(gl.GL_NORMALIZE)
                for ic, quad in enumerate(self.coords):
                    gl.glBegin(gl.GL_TRIANGLE_FAN)
                    for c in quad:
                        gl.glNormal3fv(list(self.normals[ic]))
                        gl.glColor3ubv(list(self.colors[ic]))
                        gl.glVertex3fv(list(c))
                    gl.glEnd()

        def compile(gl):
            if self.need_compile:
                gl.glNewList(self.gl_compiled_list, gl.GL_COMPILE)
                drawSolid(gl)
                gl.glEndList()
                self.need_compile = False

        def draw_list(gl):
            compile(gl)
            gl.glCallList(self.gl_compiled_list)

        sc = 0.1
        gl.glScalef(sc, sc, sc)

        draw_list(gl)

    def keyPressEvent(self, event):
        if event.key() < 256:
            ch = chr(event.key()).lower()
            if ch in ' cr':
                self.setCoords(self.sh.random_coords())
                self.need_compile = True
                self.colors = self.sh.colors()
                self.repaint()
            elif ch == 'q':
                exit(0)


class Main(QMainWindow):
    def __init__(self, sh, *args):
        super(Main, self).__init__(*args)

        self.setGeometry(100, 100, 800, 800)

        self.setCentralWidget(SHwidget(sh, self))
        self.show()


def timeIt(sh, shn):
    import time
    start_time = time.time()
    sh.coords()
    sh.normals()
    pyTime = time.time() - start_time
    print('python time:', pyTime)

    start_time = time.time()
    shn.coords()
    shn.normals()
    nbTime = time.time() - start_time
    print('numba time:', nbTime, 'ratio python/numba:', pyTime / nbTime)


if __name__ == '__main__':
    resol = 256

    # sh = SphericalHarmonics(code=random_code(), resol=resol)
    shn = SpheHarmNumba(vcode=str2floatarray(random_code()), resol=resol, colors=colorMap('Accent', resol))

    # timeIt(sh, shn)

    app = QApplication(sys.argv)
    mw = Main(shn)

    sys.exit(app.exec_())
