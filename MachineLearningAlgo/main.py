from autodiff import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def fn(x,y):
     return (x**2 + y**2)

grad = []

rrow = 50
ccol = 50

for yval in np.linspace(-5,5,rrow):
     for xval in np.linspace(-5,5,ccol):
          x = Var(name='x', value=xval)
          y = Var(name='y', value=yval)
          grad.append((xval,yval,fn(x,y).compute()))

# Plot surface
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

surf = ax.plot_surface(np.array([x[0] for x in grad]).reshape(rrow, ccol),
                       np.array([y[1] for y in grad]).reshape(rrow, ccol),
                       np.array([z[2] for z in grad]).reshape(rrow, ccol),
                       cmap = cm.coolwarm, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()