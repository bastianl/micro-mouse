from collections import OrderedDict

import numpy as np

# rotation map to record sensor data.
DIRECTIONS = OrderedDict([
    ('up', [1, 0]),
    ('right', [0, 1]),
    ('down', [-1, 0]),
    ('left', [0, -1]),
])
rotations = {'up': 0, 'right': 1, 'down': 2, 'left': 3}


def is_prev_open(current, rotation, movement, heading):
    """Check to see if the direction behind the robot is open,
    i.e. does not have a wall"""
    dirs = DIRECTIONS.keys()
    direction = direction_from_rotation(heading, rotation)
    prev_open = bfi(current)[(dirs.index(direction) - 2) % len(dirs)]
    if movement == 0:
        # is backward direction open?
        return prev_open
    else:
        # is forward direction open or, if we can't move
        # just use prev_open
        return bfi(current)[(dirs.index(direction))] or prev_open


def direction_from_rotation(heading, rotation):
    dirs = DIRECTIONS.keys()
    idx = int(dirs.index(heading) + rotation / 90) % len(dirs)
    return dirs[idx]


def rotation_from_direction(heading, direction):
    if heading == direction:
        return 0
    ds = {
        ('up', 'right'): 90,
        ('up', 'left'): -90,
        ('up', 'down'): 180,
        ('right', 'down'): 90,
        ('right', 'up'): -90,
        ('right', 'left'): 180,
        ('down', 'left'): 90,
        ('down', 'right'): -90,
        ('down', 'up'): 180,
        ('left', 'up'): 90,
        ('left', 'down'): -90,
        ('left', 'right'): 180,
    }
    return ds[(heading, direction)]


def get_updated_location(direction, location, movement=1):
    return list(np.array(location) + movement * np.array(DIRECTIONS[direction]))


def ifb(N, E, S, W):
    """int from bits
    Each value of N, E, S, W should be 0 if there is no wall,
    and 1 if there is a wall in that direction."""
    return (N * 1 + E * 2 + S * 4 + W * 8)

def bfi(n): 
    """bits from int"""
    return tuple((1 & int(n) >> i) for i in range(4))

""" Bit rotation. Taken from [0].
[0]: http://www.falatic.com/index.php/108/python-and-bitwise-rotation
"""

rol = lambda val, r_bits: \
    (val << r_bits%4) & (2**4-1) | \
    ((val & (2**4-1)) >> (4-(r_bits%4)))

ror = lambda val, r_bits: \
    ((val & (2**4-1)) >> r_bits%4) | \
    (val << (4-(r_bits%4)) & (2**4-1))