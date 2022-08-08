#!/usr/bin/env python2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import rospy
from std_msgs.msg import Int32


idx = 0
fig = plt.figure(figsize=(6, 1))


def cb(msg):
    global idx
    idx = msg.data


rospy.init_node('pngui')
rospy.Subscriber('setidx', Int32, cb)
pub = rospy.Publisher('idx', Int32, queue_size=0)


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


axnext = plt.axes([0.81, 0.25, 0.18, 0.5])
bnext = Button(axnext, 'Next')
bnext.on_clicked(nextcb)

axprev = plt.axes([0.61, 0.25, 0.18, 0.5])
bprev = Button(axprev, 'Previous')
bprev.on_clicked(prevcb)

axcur = plt.axes([0.41, 0.25, 0.18, 0.5])
bcur = Button(axcur, 'Current')
bcur.on_clicked(curcb)

axnext5 = plt.axes([0.21, 0.25, 0.18, 0.5])
bnext5 = Button(axnext5, 'Next5')
bnext5.on_clicked(next5cb)

axprev5 = plt.axes([0.01, 0.25, 0.18, 0.5])
bprev5 = Button(axprev5, 'Prev5')
bprev5.on_clicked(prev5cb)

plt.show()
