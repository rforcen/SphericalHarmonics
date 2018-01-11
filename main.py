import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow)

from SphericalHarmonics import SphericalHarmonics
from predefs import random_code
from rendererGL import RendererGL


class SHwidget(RendererGL):
    coords = None
    sh = None
    win = None

    def __init__(self, sh, win):
        super(SHwidget, self).__init__()
        self.sh = sh
        self.coords = sh.coords()
        self.win = win

        self.setFocusPolicy(Qt.StrongFocus)
        self.win.setWindowTitle('Spherical Harmonics, code:' + sh.code)

    def setCoords(self, coords):
        self.coords = coords
        self.repaint()

    def draw(self, gl):
        if self.coords is not None:
            gl.glColor3f(0.5, 0.6, 0.3)
            sc = 0.15
            gl.glScalef(sc, sc, sc)
            for quad in self.coords:
                gl.glBegin(gl.GL_LINE_LOOP)
                for c in quad:
                    gl.glVertex3fv(c)
                gl.glEnd()

    def keyPressEvent(self, event):
        if event.key() < 256:
            if chr(event.key()) == ' ':  # space bar generates random code
                self.setCoords(sh.random())
                self.win.setWindowTitle('Spherical Harmonics, code:' + sh.code)


class Main(QMainWindow):
    def __init__(self, sh, *args):
        super(Main, self).__init__(*args)

        self.setGeometry(100, 100, 800, 800)

        self.setCentralWidget(SHwidget(sh, self))
        self.show()


if __name__ == '__main__':
    sh = SphericalHarmonics(random_code())

    app = QApplication(sys.argv)
    mw = Main(sh)

    sys.exit(app.exec_())
