#!/usr/bin/env python
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray
from StringIO import StringIO
import itertools
import numpy as np


def deserialize(file, cls):
    """Deserialize and return corresponding ros message of a file.

    Args:
        file (str): serialized file location.
        cls: ros message type.

    Returns:
        The deserialized object.
    """
    buf = StringIO()
    buf.write(open(file).read())
    msg = cls()
    msg.deserialize(buf.getvalue())
    return msg


def serialize(file, cls):
    pass


def datapath(path):
    return '/home/ee904/Desktop/HuaTsai/RadarSLAM/data/' + path


def deserialize_multiarray_to_numpy(filename):
    """Deserialize and return numpy array.

    Args:
        file (str): file location.

    Returns:
        numpy.array object.
    """
    array = deserialize(filename, Float64MultiArray)
    size = [dim.size for dim in array.layout.dim]
    rangesize = [range(i) for i in size]
    strides = [dim.stride for dim in array.layout.dim]
    ret = np.zeros((size))
    for comb in itertools.product(*rangesize):
        dataidx = array.layout.data_offset
        for idx, i in enumerate(comb):
            if (idx == len(comb) - 1):
                dataidx += comb[idx]
            else:
                dataidx += comb[idx] * strides[idx + 1]
        ret[comb] = array.data[dataidx]
    return ret
