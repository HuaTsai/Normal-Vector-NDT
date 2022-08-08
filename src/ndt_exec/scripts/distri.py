#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Vector3


def pdf(x, mu, sig):
    return 1 / (sig * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x-mu) / sig) ** 2)


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2, top=0.8)
x = np.linspace(-4, 4, 1000)
y = np.array([pdf(j, 0, 1) for j in x])
# dis, = plt.plot(x, y, lw=2)
idx = 0
ax = plt.gca()


def cb(msg):
    ax.clear()
    mu = abs(msg.x)
    sig = math.sqrt(msg.y)
    dis = msg.z
    x = np.linspace(mu - 4 * sig, mu + 4 * sig, 1000)
    y = np.array([pdf(j, mu, sig) for j in x])
    ax.plot(x, y)
    ax.set_title('Correspondences #{}\n'
                 'Cell Distance: {:.4f}\n'
                 'Standardized Euclidean Distance: {:.4f}\n'.format(idx, dis, mu / sig))
    ax.set_xlim(min(-0.2, mu - 4 * sig), mu + 4 * sig)
    ax.set_ylim(-0.02, max(y) + 0.02)
    y0 = pdf(0, mu, sig)
    ax.text(0.2, y0, 'Prob: {:.4f}'.format(y0))
    ax.vlines(0, -0.02, y0, linestyles='dashed')
    ax.hlines(y0, min(-0.2, mu - 4 * sig), 0, linestyles='dashed')
    plt.draw()


rospy.init_node('distribution')
rospy.Subscriber('meancov', Vector3, cb)
pub = rospy.Publisher('idx', Int32, queue_size=0)


def nextcb(event):
    global idx
    idx = idx + 1
    pub.publish(Int32(idx))


def prevcb(event):
    global idx
    idx = idx - 1
    pub.publish(Int32(idx))


axnext = plt.axes([0.81, 0.03, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(nextcb)

axprev = plt.axes([0.7, 0.03, 0.1, 0.075])
bprev = Button(axprev, 'Previous')
bprev.on_clicked(prevcb)

plt.show()
