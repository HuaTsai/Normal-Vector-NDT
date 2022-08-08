# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

fig, ax = plt.subplots()
im = ax.imshow(harvest)

ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
ax.set_xticklabels(farmers, rotation=45, ha='right', rotation_mode='anchor')
ax.set_yticklabels(vegetables)

for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       color='w', ha='center', va='center')

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()

# %%


def heatmap(data, rows, cols, barlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="RdYlGn")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(barlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(cols, rotation=-45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(rows)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)

    fmt = matplotlib.ticker.StrMethodFormatter('{x:.1f}%')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            im.axes.text(j, i, fmt(data[i, j]),
                         ha='center', va='center', color='k')


# %%
heatmap(harvest, vegetables, farmers, 'harvest [t/year]')

# %%
