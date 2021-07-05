#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
import math
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import Vector3


id = 0
plt.ion()
fig, ax = plt.subplots()
# fig = plt.figure()
ax = fig.add_subplot(111, title='Error Distribution #?', xlabel='Error [m]', ylabel='Prob')
dis, = ax.plot([0], [0])
ax.legend()
plt.tight_layout()

def pdf(x, mu, sig):
    return 1 / (sig * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x-mu) / sig) ** 2)


def cb(msg):
    global id
    global fig
    global dis
    mu = math.sqrt(msg.x)
    sig = math.sqrt(msg.y)
    x = np.linspace(mu - 4 * sig, mu + 4 * sig, 1000)
    y = np.array([pdf(j, mu, sig) for j in x])
    dis.set_xdata(x)
    dis.set_ydata(y)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # time.sleep(0.1)


def cb2(msg):
    global id
    id = msg.data


def main():
    rospy.init_node('distribution')
    rospy.Subscriber('meancov', Vector3, cb)
    rospy.Subscriber('idx', Int32, cb2)
    rospy.spin()


if __name__ == '__main__':
    main()
