#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray

fig, ax = plt.subplots()
ax = plt.gca()


def cb(msg):
    sz = msg.layout.data_offset
    x = [i * 5 for i in range(sz)]
    esicp = []
    endt = []
    esndt = []
    for i in range(sz):
        esicp.append(msg.data[i * 3])
        endt.append(msg.data[i * 3 + 1])
        esndt.append(msg.data[i * 3 + 2])
    ax.plot(x, esicp, label='sicp', color='r')
    ax.plot(x, endt, label='ndt', color='b')
    ax.plot(x, esndt, label='sndt', color='g')
    ax.legend()
    ax.grid()
    plt.draw()


rospy.init_node('test_diffpy')
rospy.Subscriber('error', Float64MultiArray, cb)

plt.show()
