import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

def heatmap(data, rows, cols, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title)
    im = ax.imshow(data, cmap='RdYlGn', vmin=0)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('success rate [%]', rotation=-90, va='bottom')

    ax.set_xlabel('[deg]')
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(cols)
    ax.xaxis.tick_top()

    ax.set_ylabel('[m]')
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(rows, usetex=True)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)

    fmt = matplotlib.ticker.StrMethodFormatter('{x:.2f}%')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            im.axes.text(j, i, fmt(data[i, j]), ha='center', va='center', color='k')
    fig.savefig(sys.argv[2])

f = sys.argv[1]
with open(f, 'r') as file:
    lines = file.readlines()
    n = len(lines)
    title = lines[0]
rs = np.genfromtxt(f, delimiter=',', skip_header=1, skip_footer=n-2)
ts = np.genfromtxt(f, delimiter=',', skip_header=2, skip_footer=n-3)
ts = ['$\pm' + str(t) + '^\circ$' for t in ts]
data = np.genfromtxt(f, delimiter=',', skip_header=3)
heatmap(data, rs, ts, title)
