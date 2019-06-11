import cppcolormap as cmap

import matplotlib.pyplot as plt
import numpy as np

colormapNames = 'Accent,Dark2,Paired,Spectral,Pastel1,Pastel2,Set1,Set2,Set3,Blues,Greens,Greys,Oranges,Purples,Reds,BuPu,GnBu,PuBu,PuBuGn,PuRd,RdPu,OrRd,RdOrYl,YlGn,YlGnBu,YlOrRd,BrBG,PuOr,RdBu,RdGy,RdYlBu,RdYlGn,PiYG,PRGn'

# number of colors in the colormap (optional, may be omitted)
N = 128

# specify the colormap as string
cr = cmap.colormap("Reds", N)
cc = cmap.colorcycle("tue")

# or call the functions directly
c = cmap.Reds(N)
c = cmap.tue()

plt.imshow(np.reshape(cmap.colormap(colormapNames.split(',')[6], N * N) / 255., (N, N, 3)))
plt.show()
