# -*- coding: utf-8 -*-
"""
This script produces the delay analysis scheme.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors


# ================================== PLOT =====================================
fig = plt.figure(figsize=(5, 1.1), dpi=200, clear=True, facecolor='w')
ax = fig.add_axes(rect=(0.05, 0.34, 0.9, 0.35))


ax.set_xlim(-0.4, 10)
ax.set_ylim(0, 1)

ax.axis("off")
ax.tick_params(axis='y', left=False, labelleft=False)
ax.tick_params(axis='x', bottom=False, labelbottom=False)

# Time arrow
ax.annotate(
    "", xy=(ax.get_xlim()[1]-0.1, 0.),
    xytext=(ax.get_xlim()[0]+0.1, 0.),
    xycoords=ax.transData,
    arrowprops={
        'arrowstyle': "->,head_width=0.15,head_length=0.3",
        'connectionstyle': "arc3,rad=0.0",
        'facecolor': 'k',
        'shrinkA': 0,
        'shrinkB': 0,
        'lw': 1})
ax.text(ax.get_xlim()[1]+0.1, -0.2, "$t$", fontsize=11,
         ha="center", va="center", transform=ax.transData)




########## Sequence ##########

## Tweezer color gradient
white = np.array(colors.to_rgba("w"))
color1 = np.array(colors.to_rgba("red"))
raw_cmap1 = np.linspace(white, color1, 256, endpoint=True)
cmap1 = colors.ListedColormap(raw_cmap1)
## Optical pumping color gradient
color2 = np.array(colors.to_rgba("green"))
raw_cmap2 = np.linspace(white, color2, 256, endpoint=True)
cmap2 = colors.ListedColormap(raw_cmap2)


xx = np.broadcast_to(np.linspace(0., 1., 101, endpoint=True), (2, 101))
yy = np.full((2, 101), [[0], [1]])
pcolor = np.linspace(0., 1., 100, endpoint=False).reshape((1, -1))


# First rental
a1, a2 = 0.8, 3.4
ax.pcolormesh(-0.2 + (0.4+a1)*xx, yy + [[0], [0]], pcolor, cmap=cmap1)
verts = [(a1, 0.), (a1, 1.), (a2, 1.), (a2, 0.)]
poly = patches.Polygon(verts, facecolor='red', edgecolor="none")
ax.add_patch(poly)
ax.text(2, 0.48, 'Previous\nrental', ha="center", va="center", fontsize=10)


## Second rental
b1, b2 = 5.2, 8.5
verts = [(b1, 0.), (b1, 1.), (b2, 1.), (b2, 0.)]
poly = patches.Polygon(verts, facecolor='green', edgecolor="none")
ax.add_patch(poly)
ax.pcolormesh(b2 + (9.8-b2)*xx, yy, pcolor[:, -1::-1], cmap=cmap2)
ax.text(7.6, 0.48, 'Next\nrental', ha="center", va="center", fontsize=10,
        color='w')


## Checkout delay
a3 = 6.7
verts = [(a2, 0.), (a2, 1.), (a3, 1.), (a3, 0.)]
poly = patches.Polygon(verts, facecolor='red', alpha=0.4, edgecolor="none")
ax.add_patch(poly)



########## Timings ##########

arrow_props1 = {'arrowstyle': "<|-|>,head_width=0.15,head_length=0.3",
               'connectionstyle': "arc3,rad=0.0",
               'facecolor': 'k',
               'shrinkA': 0,
               'shrinkB': 0,
               'lw': 0.8}

arrow_props2 = {'arrowstyle': "->,head_width=0.14,head_length=0.28",
               'connectionstyle': "arc3,rad=0.0",
               'facecolor': 'k',
               'shrinkA': 0,
               'shrinkB': 1.5,
               'lw': 1}

ax.plot([a2, a2], [1, -0.35], '--', c="k", lw=0.8, clip_on=False)
ax.plot([b1, b1], [1.35, 0], '--', c="k", lw=0.8, clip_on=False)
ax.plot([a3, a3], [1.35, -0.35], '--', c="k", lw=0.8, clip_on=False)

ax.annotate("", xy=(a2, -0.2), xytext=(a3, -0.2), xycoords=ax.transData,
            arrowprops=arrow_props1)
ax.text((a2+a3)/2, -0.45, 'Checkout delay', fontsize=9, c="k",
          ha="center", va="center")

ax.annotate("", xy=(b1, 1.15), xytext=(a3, 1.15), xycoords=ax.transData,
            arrowprops=arrow_props1)
ax.text((a3+b1)/2, 1.5, 'Waiting\ntime', fontsize=9, c="k",
          ha="center", va="center")

# previous rental checkout
ax.annotate('', xy=(a2, -0.2), xytext=(a2-0.42, -0.44),
            xycoords=ax.transData, arrowprops=arrow_props2)
ax.text(a2-1.2, -0.58, 'Expected\ncheckout time', fontsize=9, c="k",
          ha='center', va="center")

# actual checkout
ax.annotate('', xy=(a3, -0.2), xytext=(a3+0.45, -0.43),
            xycoords=ax.transData, arrowprops=arrow_props2)
ax.text(a3+1.1, -0.55, 'Actual\ncheckout time', fontsize=9, c="k",
          ha='center', va="center")

# next rental checkin
ax.annotate('', xy=(b1, 1.2), xytext=(b1-0.55, 1.42),
            xycoords=ax.transData, arrowprops=arrow_props2)
ax.text(b1-1.5, 1.45, 'Expected\ncheckin time', fontsize=9, c="k",
          ha='center', va="center")




################################# SAVE FIGURE #################################
fig.savefig("./delay_analysis_scheme.png")
plt.show()
