import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import sys

with open(sys.argv[1], 'r') as file:
    lines = file.readlines()

x, icp, nicp, sicp, pndt, dndt, sndt = [[float(i) for i in j.split(',')] for j in lines]

ms = 3
fig, ax = plt.subplots()
ax.loglog(x, icp, 'o-', label='point-to-point ICP', markersize=ms)
ax.loglog(x, nicp, 'o-', label='point-to-plane ICP', markersize=ms)
ax.loglog(x, sicp, 'o-', label='symmetric ICP', markersize=ms)
ax.loglog(x, pndt, 'o-', label='point-to-distribution NDT', markersize=ms)
ax.loglog(x, dndt, 'o-', label='distribution-to-distribution NDT', markersize=ms)
ax.loglog(x, sndt, 'o-', label='symmetric NDT', markersize=ms)
# ax.plot(x, icp, 'o-', label='point-to-point ICP', markersize=ms)
# ax.plot(x, nicp, 'o-', label='point-to-plane ICP', markersize=ms)
# ax.plot(x, sicp, 'o-', label='symmetric ICP', markersize=ms)
# ax.plot(x, pndt, 'o-', label='point-to-distribution NDT', markersize=ms)
# ax.plot(x, dndt, 'o-', label='distribution-to-distribution NDT', markersize=ms)
# ax.plot(x, sndt, 'o-', label='symmetric NDT', markersize=ms)

ax.set_xlabel('Error before ICP/NDT iteration [m]')
ax.set_ylabel('Error after ICP/NDT iteration [m]')
ax.grid()
ax.legend(loc='upper left')
ax.set_xlim([0.01, 1])
ax.set_ylim([0, 1])
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
fig.savefig(sys.argv[2])