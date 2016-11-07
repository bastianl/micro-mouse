from __future__ import print_function, division
from copy import copy
import logging

import numpy as np

from utils import DIRECTIONS, rotations
from utils import ifb, bfi, rol
from utils import get_updated_location
from utils import is_prev_open
from utils import rotation_from_direction
from utils import direction_from_rotation
from utils import direction_to_square
from utils import in_goal

from heuristics import Random, AStar


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
        self.reached_goal = False
        # is the previous square open?
        self.prev_open = 0
        # heuristic function we use to weight possible moves in mapping
        self.heuristic = AStar(self)
        logging.debug("Using heuristic {}".format(self.heuristic))
        # step counter
        self.step = 0
        # maximum steps used to map
        self.STEP_LIMIT = 900
        self.moves = []

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

        state = self._map[self.location[0], self.location[1]]
        if state != 0:
            global_state = state
        else:
            S = self.prev_open
            W, N, E = np.array(sensors) > 0
            # adjust the state based on heading direction
            global_state = rol(ifb(N, E, S, W), rotations[self.heading])


        if self.mapping:
            if not state:
                self.record_state(self.location, global_state)
            if in_goal(self.location, self.maze_dim):
                self.reached_goal = True
                self.goal_location = self.location
                self.heuristic.reverse_heuristic()

            if self.reached_goal and self.done_exploring():
                self.location = [0, 0]
                self.heading = 'up'
                self.mapping = False
                self.path_to_goal = self.heuristic.get_best_path(self.goal_location)
                return "Reset", "Reset"

            rotation, movement = self.get_next_mapping_move(bfi(global_state))
        else: 
            path = self.path_to_goal
            target = path.pop()
            direction = direction_to_square(target, self.location)
            rotation = rotation_from_direction(self.heading, direction)
            movement = 1
            # move more squares, if possible
            while len(path) > 0 and direction_to_square(path[-1], target) == direction \
                    and movement < 3:
                target = path.pop()
                movement += 1

        old_loc = copy(self.location)

        self.update_location(rotation, movement)

        self.moves.append([old_loc, self.location, rotation, movement])

        return rotation, movement

    def record_state(self, location, state):
        """Record the state at a given location."""
        assert isinstance(state, int)
        loc = list(location)

        # loc[0] = (self.maze_dim - 1) - loc[0]
        if self._map[loc[0], loc[1]] == 0:
            # only update state if we haven't been there!
            # this allows us to turn freely
            logging.debug("Recording state for {}: {}".format(loc, state))
            self._map[loc[0], loc[1]] = state

    def get_next_mapping_move(self, global_state):
        open_states = []
        for direction, state in zip(DIRECTIONS.keys(), global_state):
            if state > 0:
                open_states.append([direction])

        movement = 1

        # always call the heuristic function!
        # some heuristics (A*) need to keep track of state
        direction = self.heuristic.get_move(open_states)

        rotation = rotation_from_direction(self.heading, direction)

        if rotation == 180:
            # no reverse movement to keep it simple
            rotation = 90
            movement = 0

        # calculate whether the next position will be open
        # get the next location given direction, and movement
        current = self._map[self.location[0], self.location[1]]
        self.prev_open = is_prev_open(current, rotation, movement, self.heading)
        logging.debug("Prev open: {}".format(self.prev_open))

        return rotation, movement


    def update_location(self, rotation, movement):
        movement = max(min(int(movement), 3), -3) # fix to range [-3, 3]

        direction = direction_from_rotation(self.heading, rotation)

        self.location = get_updated_location(direction, self.location, movement)
        self.heading = direction

        logging.debug("Moving: {} Rotating: {}. New Location: {}. Heading {}".format(
            movement, rotation, self.location, self.heading))


    def done_exploring(self):
        sq = self.maze_dim ** 2
        ratio = (self._map > 0).sum() / sq
        import ipdb; ipdb.set_trace()
        return ratio >= 0.6 or self.step >= sq
