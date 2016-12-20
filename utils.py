from collections import OrderedDict

import numpy as np

# rotation map to record sensor data.
DIRECTIONS = OrderedDict([
    ('up', np.array([1, 0])),
    ('right', np.array([0, 1])),
    ('down', np.array([-1, 0])),
    ('left', np.array([0, -1])),
])
rotations = {'up': 0, 'right': 1, 'down': 2, 'left': 3}


class State(object):
    """State container object for the robot. Stores information
    on where the robot currently is, and what it senses."""

    def __init__(self, location, heading, sensors, prev_open):
        S = prev_open
        W, N, E = sensors
        # adjust the sensors based on the heading direction
        sensors = rotate([N, E, S, W], rotations[heading])
        self.sensors = sensors
        self._loc = location

    @property
    def state(self):
        return ifb(*list(np.int64(np.array(self.sensors) > 0)))

    @property
    def location(self):
        return np.array(self._loc)

    @property
    def open_squares(self):
        open_states = []
        for direction, state in zip(DIRECTIONS.keys(), self.sensors):
            if state > 0:
                open_states.append(direction)
        return open_states

    @property
    def visible_squares(self):
        """Use the sensor data to calculate visible squares"""
        visible = []
        for idx, val in enumerate(self.sensors):
            direction = DIRECTIONS.values()[idx]
            while val > 0:
                visible.append(self.location + direction * val)
                val -= 1
        return visible

    def __repr__(self):
        return '<Location {} - seeing: {}>'.format(self.location, self.sensors)


class Map(object):

    def __init__(self, dim):
        self.dim = dim
        self._data = np.zeros([dim, dim], dtype='int64')

    def __call__(self, location):
        return self._data[location[0], location[1]]

    def bit(self, loc):
        return bfi(self(loc))

    def set(self, location, state):
        assert isinstance(state, int)
        self._data[location[0], location[1]] = state

    def ratio(self):
        return (self._data > 0).sum() / float(self.dim ** 2)

    def data(self):
        return self._data.copy()

    def __repr__(self):
        return str(self._data)


def path_to_point(maze, point_a, point_b):
    """Given a partial map, return the set of legal moves
    that would result in a path from one point to another.
    Uses the expansion grid method.
    Note: while this is a similar implementation to our A*
    mapping method, it is different in that the robot
    can only expand one location at a time. Here we expand
    all adjacent explored squares in the maze.
    Note: Path does not return point_a, but it does return point_b as
    first point. Get the next point by calling path.pop(), or path[-1]."""
    maze = maze._data.copy()
    def not_in(loc, square_list):
        return all([dist(loc, x[0]) > 0 for x in square_list])
    dim = len(maze)
    g_values = np.zeros([dim, dim], dtype='int64')
    target = [point_a]
    closed_squares = []
    open_squares = []
    reachable = False
    while dist(target[0], point_b) != 0 and not reachable:
        closed_squares.append(target)
        loc = target[0]
        g_value = g_values[loc[0], loc[1]] + 1
        states = bfi(maze[loc[0], loc[1]])
        for idx, state in enumerate(states):
            if state:
                loc = get_updated_location(DIRECTIONS.keys()[idx], target[0])
                if not_in(loc, open_squares) and not_in(loc, closed_squares) \
                        and maze[loc[0], loc[1]] > 0:
                    h = dist(loc, point_b) # heuristic function
                    f = g_value + h
                    open_squares.append([loc, f, h, g_value])
                    g_values[loc[0], loc[1]] = g_value
        # expand next square with the lowest f value
        open_squares = sorted(open_squares, key=lambda x: x[1], reverse=True)
        target = open_squares.pop()
        if dist(target[0], point_b) == 1:
            # if within one square we test to see
            # if there is a wall between the two squares.
            direction = direction_to_square(point_b, target[0])
            idx = DIRECTIONS.keys().index(direction)
            loc = target[0]
            reachable = bfi(maze[loc[0], loc[1]])[idx]

    return path_from_g_values(maze, g_values, point_a, point_b)


def path_from_g_values(maze, g_values, point_a, point_b):
    path = [point_b]

    while True:
        loc = path[-1]
        if g_values[loc[0], loc[1]] == 1:
            # we are within 1 square, so break
            # Don't append point_a. This is not useful for the robot,
            # as it typically starts on that square.
            break
        states = bfi(maze[loc[0], loc[1]])
        open_squares = []
        for idx, state in enumerate(states):
            if state:
                p = get_updated_location(DIRECTIONS.keys()[idx], loc)
                g_value = g_values[p[0], p[1]]
                if g_value > 0:
                    open_squares.append([p, g_value])
        open_squares = sorted(open_squares, key=lambda x: x[1], reverse=True)
        path.append(open_squares.pop()[0])
    return path



def is_prev_open(current, rotation, movement, heading):
    """Check to see if the direction behind the robot is open,
    i.e. does not have a wall. This can be infered based off 
    of the movement of the robot."""
    dirs = DIRECTIONS.keys()
    direction = direction_from_rotation(heading, rotation)
    behind_open = bfi(current)[(dirs.index(direction) - 2) % len(dirs)]
    if movement == 0:
        # i'm facing a new direction. in that direction, is the
        # backward wall open?
        return behind_open
    else:
        # is forward direction open or, if we can't move
        # just use the behind direction
        # we know that if the robot can move, the spot behind it
        # will be open when it is in the next square, as thats
        # where it came from!
        return bfi(current)[(dirs.index(direction))] or behind_open


def direction_to_square(to, cur):
    assert dist(to, cur) == 1
    direction = [to[0] - cur[0], to[1] - cur[1]] 
    keys = [ key for key, value in DIRECTIONS.iteritems() if np.all(value == direction) ]
    return keys[0]


def direction_from_rotation(heading, rotation):
    dirs = DIRECTIONS.keys()
    idx = int(dirs.index(heading) + rotation / 90) % len(dirs)
    return dirs[idx]


def rotation_from_direction(heading, direction):
    keys = DIRECTIONS.keys()
    angle = ((keys.index(direction) - keys.index(heading)) * 90) % 360

    if angle == 270:
        angle = -90

    return angle


def in_goal(loc, dim):
    bounds = [dim / 2 - 1, dim / 2]
    return loc[0] in bounds and loc[1] in bounds


def get_updated_location(direction, location, movement=1):
    """New location, after movement in a direction."""
    return list(np.array(location) + movement * np.array(DIRECTIONS[direction]))


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def rotate(l, n):
    """Rotate a list"""
    return l[-n:] + l[:-n]


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


if __name__ == "__main__":
    # test helper functions
    assert ifb(*bfi(10)) == 10
    assert rol(ror(10, 1), 1) == 10
    assert rol(10, 0) == 10
    assert rotation_from_direction('up', 'down') == 180
    assert rotation_from_direction('down', 'left') == 90
    assert rotation_from_direction('right', 'up') == -90
    assert rotation_from_direction('down', 'up') == 180 
    assert rotation_from_direction('right', 'left') == 180 
    assert rotation_from_direction('left', 'up') == 90
    assert direction_from_rotation('up', 90) == 'right'
    assert direction_from_rotation('down', -90) == 'right'
    assert direction_from_rotation('left', -90) == 'down'
    assert direction_from_rotation('left', 90) == 'up'
    assert direction_from_rotation('left', 180) == 'right'
    assert is_prev_open(9, 90, 0, 'up')
    assert not is_prev_open(1, 0, 0, 'up')
    assert is_prev_open(1, 0, 1, 'up')
    assert not is_prev_open(1, 90, 0, 'up')
    assert is_prev_open(1, 0, 0, 'down')
    assert not is_prev_open(14, 90, 0, 'right')
    assert direction_to_square([1, 0], [0, 0]) == 'up'
    assert direction_to_square([7, 1], [7, 0]) == 'right'
    assert direction_to_square([8, 5], [9, 5]) == 'down'
    assert direction_to_square([3, 9], [3, 10]) == 'left'
    print("All tests passed!")