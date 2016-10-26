from __future__ import print_function, division
from copy import copy
from collections import OrderedDict
from random import random
import numpy as np

# rotation map to record sensor data.
DIRECTIONS = OrderedDict([
    ('up', [1, 0]),
    ('right', [0, 1]),
    ('down', [-1, 0]),
    ('left', [0, -1]),
])
rotations = {'up': 0, 'right': 1, 'down': 2, 'left': 3}


class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''

        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim
        # the robots map of the maze
        # the robots goal in the mapping stage is to
        # fill every 0 in the map with a value describing
        # the walls on each of the 4 sides of the square.
        self._map = np.zeros([maze_dim, maze_dim], dtype='int64')
        # are we in mapping mode?
        self.mapping = True
        # is the previous square open?
        self.prev_open = 0
        # heuristic function we use to weight possible moves in mapping
        self.mapping_heuristic = 'a_star'
        # step counter
        self.step = 0
        # maximum steps used to map
        self.STEP_LIMIT = 900

    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''
        self.step += 1

        rotation = 0
        movement = 0

        if self.mapping:
            # TODO: this is too naive. lets use the previously
            # explored square if possible
            S = self.prev_open
            W, N, E = np.array(sensors) > 0
            state = ifb(N, E, S, W)
            global_state = rol(copy(state), rotations[self.heading])
            self.record_state(self.location, global_state)

            if np.all(self._map > 0) or self.step > self.STEP_LIMIT:
                if self.in_goal():
                    self.mapping = False
                    return "Reset", "Reset"
                else:
                    # move towards goal given our map knowledge
                    import ipdb; ipdb.set_trace()
            else:
                rotation, movement = self.get_next_mapping_move(bfi(global_state))

        else:
            # map should be complete. run fastest route
            pass

        self.update_location(rotation, movement)

        # if self._map[self.location[0], self.location[1]] == 0:
        # print("new location!")

        return rotation, movement

    def record_state(self, location, state):
        """Record the state at a given location."""
        assert isinstance(state, int)
        loc = list(location)
        # loc[0] = (self.maze_dim - 1) - loc[0]
        if self._map[loc[0], loc[1]] == 0:
            # only update state if we haven't been there!
            # this allows us to turn freely
            print("Recording state for {}: {}".format(loc, state))
            self._map[loc[0], loc[1]] = state

    def get_next_mapping_move(self, global_state):
        open_states = []
        for direction, state in zip(DIRECTIONS.keys(), global_state):
            if state > 0:
                open_states.append([direction])
        if len(open_states) == 0:
            # we've reached a dead end!!
            rotation, movement = 90, 0
        elif len(open_states) > 1:
            for st in open_states:
                loc = get_updated_location(st[0], self.location)
                try:
                    heuristic = self.get_heuristic(loc, len(open_states))
                except IndexError as e:
                    import ipdb; ipdb.set_trace()
                st.append(heuristic)

            direction, heuristic = sorted(open_states, key=lambda x: x[1],
                                          reverse=True)[0]
            print("Best heuristic out of {} choices: {}".format(
                len(open_states), heuristic))
        else:
            direction = open_states[0][0]

        rotation = rotation_from_direction(self.heading, direction)

        if rotation == 180:
            # can't turn 180 degrees, must do it in steps
            rotation, movement = 90, 0

        if rotation > 0:
            movement = 0
        else:
            movement = 1

        # calculate whether the next position will be open
        # get the next location given direction, and movement
        current = self._map[self.location[0], self.location[1]]
        self.prev_open = is_prev_open(current, rotation, movement, self.heading)
        print("Prev open: {}".format(self.prev_open))

        return rotation, movement


    def update_location(self, rotation, movement):
        movement = max(min(int(movement), 3), -3) # fix to range [-3, 3]

        direction = direction_from_rotation(self.heading, rotation)

        self.location = get_updated_location(direction, self.location, movement)
        self.heading = direction

        print("Moving: {} Rotating: {}. New Location: {}. Heading {}".format(
            movement, rotation, self.location, self.heading))

    def get_heuristic(self, location, num_states):
        if self.mapping_heuristic == 'random':
            return random_heuristic(location, num_states, self._map)
        else:
            return a_star_heuristic(location, self.maze_dim)

    def in_goal(self):
        d = self.maze_dim // 2
        return d - 1 <= self.location[0] <= d and \
            d - 1 <= self.location[1] <= d


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


def random_heuristic(loc, num_states, maze, weight=5):
    """Randomly provide a heuristic for a node.
    Weight is the multiplier for preference if that node
    has not yet been explored."""
    # prefer unexplored states
    mult = weight * (maze[loc[0], loc[1]] == 0) + 1
    return random() / num_states * (mult)


def a_star_heuristic(loc, dim):
    """Get an optimistic prediction to the goal"""
    [x, y] = loc
    # to account for the 2x2 goal in the center
    x_dim = dim // 2 - 1 if x < dim // 2 else dim //2
    y_dim = dim // 2 - 1 if y < dim // 2 else dim //2
    return abs(x - x_dim) + abs(y - y_dim)


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
    print("All tests passed!")
