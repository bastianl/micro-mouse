from __future__ import print_function, division
from copy import copy

import numpy as np

from utils import DIRECTIONS, rotations
from utils import ifb, bfi, rol
from utils import get_updated_location
from utils import is_prev_open
from utils import rotation_from_direction
from utils import direction_from_rotation

from heuristics import RandomHeuristic


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
        self.mapping_heuristic = RandomHeuristic(self)
        print("Using heuristic {}".format(self.mapping_heuristic))
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
        else:
            # note, even if there is only one open state the
            # heuristic algorithm may want to turn around
            direction = self.mapping_heuristic.get_move(open_states)

        rotation = rotation_from_direction(self.heading, direction)

        if rotation == 180:
            # can't turn 180 degrees, must do it in steps
            rotation, movement = 90, 0

        # only ever get the next 
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

    def in_goal(self):
        d = self.maze_dim // 2
        return d - 1 <= self.location[0] <= d and \
            d - 1 <= self.location[1] <= d


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
