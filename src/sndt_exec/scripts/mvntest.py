# %%
import matplotlib
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import sqrt
from math import pi
from mpl_toolkits import mplot3d
%matplotlib inline


def Rad(deg):
    return deg * pi / 180


def plot_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x, mean_y = mean
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def Rt(x, y, th):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.array([[x], [y]])
    return R, t


def LS(cost):
    return 0.5 * cost ** 2


def ICP(p, q, x, y, th):
    R, t = Rt(x, y, th)
    cost = R.dot(p) + t - q
    return LS(cost)


def NICP(p, q, nq, x, y, th):
    R, t = Rt(x, y, th)
    cost = (R.dot(p) + t - q).T.dot(nq)
    return LS(cost)


def SICP(p, np, q, nq, x, y, th):
    R, t = Rt(x, y, th)
    cost = (R.dot(p) + t - q).T.dot(np + nq)
    return LS(cost)


def P2D(up, uq, cq, x, y, th):
    R, t = Rt(x, y, th)
    m = R.dot(up) + t - uq
    cost = sqrt(m.T.dot(np.linalg.inv(cq).dot(m)))
    return LS(cost)


def D2D(up, cp, uq, cq, x, y, th):
    R, t = Rt(x, y, th)
    m = R.dot(up) + t - uq
    c = R.dot(cp).dot(R.T) + cq
    cost = sqrt(m.T.dot(np.linalg.inv(c).dot(m)))
    return LS(cost)


def SNDT(up, cp, unp, cnp, uq, cq, unq, cnq, x, y, th):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.array([[x], [y]])
    if (unp.T.dot(unq)[0, 0] < 0):
        unp = -unp
    m1 = R.dot(up) + t - uq
    m2 = R.dot(unp) + unq
    c1 = R.dot(cp).dot(R.T) + cq
    c2 = R.dot(cnp).dot(R.T) + cnq
    mu = m1.T.dot(m2)[0, 0]
    std = sqrt(m1.T.dot(c2.dot(m1)) + m2.T.dot(c1.dot(m2)) + np.trace(c1.dot(c2)))
    cost = mu / std
    return LS(cost)


def SNDT2(up, cp, unp, uq, cq, unq, x, y, th):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.array([[x], [y]])
    if (unp.T.dot(unq)[0, 0] < 0):
        unp = -unp
    m1 = R.dot(up) + t - uq
    m2 = R.dot(unp) + unq
    c1 = R.dot(cp).dot(R.T) + cq
    mu = m1.T.dot(m2)[0, 0]
    std = sqrt(m2.T.dot(c1.dot(m2)))
    cost = mu / std
    return LS(cost)


def SNDTMuStd(up, cp, unp, cnp, uq, cq, unq, cnq, x, y, th):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.array([[x], [y]])
    m1 = R.dot(up) + t - uq
    m2 = R.dot(unp) + unq
    c1 = R.dot(cp).dot(R.T) + cq
    c2 = R.dot(cnp).dot(R.T) + cnq
    mu = m1.T.dot(m2)[0, 0]
    std = sqrt(m1.T.dot(c2.dot(m1)) + m2.T.dot(c1.dot(m2)) + np.trace(c1.dot(c2)))
    # std = sqrt(m2.T.dot(c1.dot(m2)) + np.trace(c1.dot(c2)))
    return mu, std


# %%
# Data
up = np.array([[-30.638454], [64.600220]])
cp = np.array([[0.191993, 0.177331], [0.177331, 0.170498]])
unp = np.array([[-0.597271], [0.790676]])
cnp = np.array([[0.013271, 0.009066], [0.009066, 0.006337]])
uq = np.array([[-16.883654], [58.616450]])
cq = np.array([[0.191993, 0.177331], [0.177331, 0.170498]])
unq = np.array([[-0.597271], [0.790676]])
cnq = np.array([[0.013271, 0.009066], [0.009066, 0.006337]])

# %%
def Distribution(up, cp, unp, cnp, uq, cq, unq, cnq, x, y, th):
    R, t = Rt(x, y, th)
    up2 = R.dot(up) + t
    unp2 = R.dot(unp)
    cp2 = R.dot(cp).dot(R.T)
    cnp2 = R.dot(cnp).dot(R.T)
    n = 10000
    dp = np.random.multivariate_normal(up2.flatten(), cp2, n).T
    dq = np.random.multivariate_normal(uq.flatten(), cq, n).T
    dnp = np.random.multivariate_normal(unp2.flatten(), cnp2, n).T
    dnq = np.random.multivariate_normal(unq.flatten(), cnq, n).T
    costs = []
    for i in range(n):
        costs.append((dp[:, i] - dq[:, i]).dot(dnp[:, i] + dnq[:, i]))
    fig, ax = plt.subplots()
    ax.hist(costs, bins=100, density=True, alpha=0.6, color='b', label='Real')
    return ax


def Distribution2(up, cp, unp, uq, cq, unq, x, y, th):
    R, t = Rt(x, y, th)
    up2 = R.dot(up) + t
    cp2 = R.dot(cp).dot(R.T)
    n = 10000
    dp = np.random.multivariate_normal(up2.flatten(), cp2, n).T
    dq = np.random.multivariate_normal(uq.flatten(), cq, n).T
    costs = []
    for i in range(n):
        costs.append((dp[:, i] - dq[:, i]).dot(unp + unq)[0])
    fig, ax = plt.subplots()
    ax.hist(costs, bins=100, density=True, alpha=0.6, color='b', label='Real')
    return ax


# Distribution(up, cp, unp, cnq, uq, cq, unq, cnq, 0, 0, 0)
ax = Distribution(up, cp, unp, cnq, uq, cq, unq, cnq, 112.881, -1.76242, Rad(-90))
# mu, std = SNDTMuStd(up, cp, unp, cnq, uq, cq, unq, cnq, 112.881, -1.76242, Rad(-90))
# x = np.linspace(mu - 4 * std, mu + 4 * std, 100)
# y = norm.pdf(x, mu, std)
# ax.plot(x, y, 'g', linewidth=2, label='Theory')
# ax.legend()
Distribution2(up, cp, unp, uq, cq, unq, 112.881, -1.76242, Rad(-90))

# %%
origx, origy = 13.7548, -5.98377
gap = 10
xx = np.linspace(origx - gap, origx + gap, 100)
yy = np.linspace(origy - gap, origy + gap, 100)
# tt = np.linspace(-0.25, 0.25, 5)
tt = [0]
x, y = np.meshgrid(xx, yy)

# %%
z = np.zeros(x.shape)
for t in tt:
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # z[i, j] = SNDT(up, cp, unp, cnp, uq, cq, unq, cnq, x[i, j], y[i, j], t)
            z[i, j] = SNDT2(up, cp, unp, uq, cq, unq, x[i, j], y[i, j], t)
    fig, ax = plt.subplots()
    ax.contourf(x, y, z, 50, cmap='coolwarm')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, cmap='coolwarm', edgecolor='none')

# %%
origx, origy = 13.7548, -5.98377
gap = 10
xx = np.linspace(origx - gap, origx + gap, 100)
yy = [0]
tt = np.linspace(-0.25, 0.25, 50)
x, t = np.meshgrid(xx, tt)
z = np.zeros(x.shape)

for y in yy:
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # z[i, j] = SNDT(up, cp, unp, cnp, uq, cq, unq, cnq, x[i, j], y, t[i, j])
            z[i, j] = SNDT2(up, cp, unp, uq, cq, unq, x[i, j], y, t[i, j])
    fig, ax = plt.subplots()
    ax.contourf(x, t, z, 50, cmap='coolwarm')
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, t, z, cmap='coolwarm', edgecolor='none')

# %%
origx, origy = 13.7548, -5.98377
gap = 10
xx = [0]
yy = np.linspace(origx - gap, origx + gap, 100)
tt = np.linspace(-0.25, 0.25, 100)
y, t = np.meshgrid(yy, tt)
z = np.zeros(y.shape)

for x in xx:
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            z[i, j] = SNDT(up, cp, unp, cnp, uq, cq, unq, cnq, x, y[i, j], t[i, j])
            # z[i, j] = SNDT2(up, cp, unp, uq, cq, unq, x, y[i, j], t[i, j])
    fig, ax = plt.subplots()
    # ax = plt.axes(projection='3d')
    ax.contourf(y, t, z, 100, cmap='jet')

    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(211, projection='3d')
    ax.view_init(45, 45)
    ax.plot_surface(y, t, z, cmap='jet', edgecolor='none')
    ax = fig.add_subplot(212, projection='3d')
    ax.view_init(45, 135)
    ax.plot_surface(y, t, z, cmap='jet', edgecolor='none')

# %%
