#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Vector3

idx = 0
ite = 0
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2, top=0.8)
ax = plt.gca()


def cb(msg):
    global idx
    idx = msg.data


def cb2(msg):
    er = msg.x
    et = msg.y
    ax.set_title('Index #{}: err ({}, {})\n'.format(idx, er, et))
    plt.draw()



rospy.init_node('pngui')
rospy.Subscriber('setidx', Int32, cb)
rospy.Subscriber('err', Vector3, cb2)
pub = rospy.Publisher('idx', Int32, queue_size=0)
pub2 = rospy.Publisher('iter', Int32, queue_size=0)


def posttask(offset):
    global idx
    idx += offset
    pub.publish(Int32(idx))


def nextcb(event):
    posttask(1)


def prevcb(event):
    posttask(-1)


def curcb(event):
    posttask(0)


def next5cb(event):
    posttask(5)


def prev5cb(event):
    posttask(-5)


def iter0cb(event):
    global ite
    ite = 0
    pub2.publish(Int32(ite))


def iterncb(event):
    global ite
    ite += 1
    pub2.publish(Int32(ite))


axnext = plt.axes([0.81, 0.03, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(nextcb)

axprev = plt.axes([0.7, 0.03, 0.1, 0.075])
bprev = Button(axprev, 'Previous')
bprev.on_clicked(prevcb)

axcur = plt.axes([0.59, 0.03, 0.1, 0.075])
bcur = Button(axcur, 'Current')
bcur.on_clicked(curcb)

axnext5 = plt.axes([0.48, 0.03, 0.1, 0.075])
bnext5 = Button(axnext5, 'Next5')
bnext5.on_clicked(next5cb)

axprev5 = plt.axes([0.37, 0.03, 0.1, 0.075])
bprev5 = Button(axprev5, 'Prev5')
bprev5.on_clicked(prev5cb)

axitern = plt.axes([0.26, 0.03, 0.1, 0.075])
bitern = Button(axitern, 'Itern')
bitern.on_clicked(iterncb)

axiter = plt.axes([0.15, 0.03, 0.1, 0.075])
biter0 = Button(axiter, 'Iter0')
biter0.on_clicked(iter0cb)

plt.show()
