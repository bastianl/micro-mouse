from __future__ import print_function, division

from random import random
import logging
from operator import itemgetter

import numpy as np

from utils import DIRECTIONS
from utils import get_updated_location
from utils import dist
from utils import direction_to_square
from utils import path_to_point
from utils import direction_from_rotation
from utils import rotation_from_direction
from utils import path_from_g_values
from utils import in_goal
from utils import Map


class BaseHeuristic(object):
    """Base class for heuristic objects. Stores whole robot
    action so we can the many properties on the robot to determine
    the best move."""

    def __init__(self, robot):
        self.robot = robot

    def get_move(self, open_states):
        robot = self.robot
        for st in open_states:
            loc = get_updated_location(st, robot.location)
            try:
                heuristic = self.heuristic(loc)
            except IndexError as e:
                import ipdb; ipdb.set_trace()
            st.append(heuristic)

        move = self.best_move(open_states)
        # some heuristics keep track of state
        self.record_move(move, open_states)

        return move

    def best_move(self, open_states):
        vals = sorted(open_states, key=itemgetter(1),
                      reverse=True)
        if len(vals) == 0:
            # we've reached a dead end!!
            logging.debug("Dead end! {}".format(self.robot.location))
            return direction_from_rotation(self.robot.heading, 90)

        direction, heuristic = vals[0]
        logging.debug("Best heuristic out of {} choices: {}".format(
            len(open_states), heuristic))
        return direction

    def heuristic(self, loc, open_states):
        raise NotImplemented()

    def record_move(self, move, open_states):
        return


class Random(BaseHeuristic):
    """Random heuristic. Make decisions randomly, but with a
    weighting factor to determine preference towards squares
    that have not been explored."""

    def __init__(self, robot, weight=5):
        self.robot = robot
        self.weight = weight

    def heuristic(self, loc, open_states):
        """Randomly provide a heuristic for a node.
        Weight is the multiplier for preference if that node
        has not yet been explored."""
        # prefer unexplored states
        num_states = len(open_states)
        maze = self.robot.map
        mult = self.weight * (maze[loc[0], loc[1]] == 0) + 1
        return random() / num_states * (mult)


class AStar(BaseHeuristic):

    def __init__(self, *args, **kwds):
        BaseHeuristic.__init__(self, *args, **kwds)
        # g-values for A*
        self.g_values = Map(self.robot.maze_dim)
        self.g_values.set([0, 0], 1)
        # parameters used later in simulation
        self.reset = {}
        self.TURN_PENALTY = 2
        self.reverse = False

    def get_move(self, state):
        robot = self.robot
        states = []
        for st in state.open_squares:
            loc = get_updated_location(st, robot.location)
            states.append(self.heuristic(loc, direction=st))

        robot = self.robot
        # remove moves we have already been to in open states
        def not_explored(state):
            loc = state.location
            return robot.map(loc) == 0

        states = list(filter(not_explored, states))

        # sort by lowest f_value
        states = sorted(states, key=lambda x: x.f_value,
                        reverse=False)

        reset = self.reset

        if not reset and len(states) == 0:
            # no unexplored squares.
            next_square = self.best_unexplored_square()
            logging.debug("Stuck, reseting to {}".format(next_square))
            path = path_to_point(robot.map, robot.location, next_square)
            reset = dict(path=path)

        if reset and len(reset['path']) == 0:
            # if we made it to reset point
            reset = dict()
            logging.debug("Made it to reset point!")

        if reset:
            next_square = reset['path'].pop()
            direction = direction_to_square(next_square, robot.location)
            # it may take two squares for robot to get to location..
            # in that case we need to keep the relevant square until
            # robot can get there..
            if rotation_from_direction(robot.heading, direction) == 180:
                reset['path'].append(next_square)
        else:
            if len(states) == 0:
                # dead end, turn around.
                direction = direction_from_rotation(robot.heading, 180)
            else:
                direction, heuristic = states[0].direction, states[0].f_value
                logging.debug("Best heuristic out of {} choices: {}".format(
                    len(states), heuristic))
                # some heuristics keep track of state
                self.record_move(direction, state)

        self.reset = reset

        return direction

    def heuristic(self, loc, direction=None, distance=None):
        """Get an optimistic prediction to the goal"""
        dim = self.robot.maze_dim
        [x, y] = loc
        # to account for the 2x2 goal in the center
        x_dim = dim // 2 - 1 if x < dim // 2 else dim // 2
        y_dim = dim // 2 - 1 if y < dim // 2 else dim // 2
        if self.reverse:
            h_value = dist([x, y], [0, 0])
        else:
            h_value = dist([x, y], [x_dim, y_dim])
        g_value = self.g_values(loc)
        return _AStarState(direction=direction,
                           distance=distance,
                           location=loc, 
                           g_value=g_value, 
                           h_value=h_value)

    def record_move(self, move, state):
        # A* expansion step: record g-values
        assert move in state.open_squares
        loc = self.robot.location
        for square in state.visible_squares:
            g_val = self.g_values(loc) + dist(loc, square)
            # TODO: incorperate turn penalty!

            if self.g_values(square) == 0:
                self.g_values.set(square, g_val)

            if g_val < self.g_values(square):
                # backpropagate g-values
                # we have found a more efficient route to a square
                open_squares = [[square, g_val]]
                while True:
                    # TODO: this might be broken..
                    for square, g_val in open_squares:
                        self.g_values.set(square, g_val)
                    g_val += 1
                    open_squares = []
                    for idx, state in enumerate(self.robot.map.bit(square)):
                        # get new_location
                        new_loc = get_updated_location(DIRECTIONS.keys()[idx], square)
                        if state > 0 and self.g_values(new_loc) > (g_val):
                            open_squares.append([new_loc, g_val])
                    if len(open_squares) == 0:
                        break

    def best_unexplored_square(self):
        maze = self.robot.map
        loc = self.robot.location
        open_list = []
        for i in range(maze.dim):
            for j in range(maze.dim):
                g_val = self.g_values([i, j])
                # TODO: can only select squares that
                # we also know how to get to...
                if maze([i, j]) == 0 and g_val > 0:
                    state = self.heuristic([i, j],
                                           distance=dist([i, j], loc))
                    open_list.append(state)

        def open_squares(loc, maze):
            values = []
            for v in DIRECTIONS.values():
                d = loc + np.array(v)
                # criterium for a closed direction, or
                # a space we have already visited.
                if np.any(d >= dim) or maze(d) > 0:
                    values.append(d)
            return values

        def is_cut_off(state):
            """Use the expansion method to see if we can
            reach the goal from this square or not. If not, 
            return False so we don't continue to attempt
            exploring this area."""

            open_list = [state.location]
            closed_list = []

            while len(open_list) > 0:
                loc = open_list.pop()
                closed_list.append(loc)

                if in_goal(loc, maze.dim):
                    return True

                for v in DIRECTIONS.values():
                    d = loc + np.array(v)
                    # criterium for a closed direction, or
                    # a space we have already visited.
                    if np.all(d > 0) and np.all(d < maze.dim) and maze(d) == 0 \
                            and not any(dist(d, z) == 0 for z in closed_list):
                        open_list.append(d)

            return False

        if not self.reverse:
            open_list = list(filter(is_cut_off, open_list))

        # return the closest unexplored square
        def best(state):
            # return state.f_value
            # return state.f_value / 10.0 + state.distance
            return state.f_value / 15.0 + state.distance

            # return float(state.f_value) / 10.0 + state.distance
        square = sorted(open_list, key=best)[0]
        loc = square.location
        g_val = self.g_values(loc)
        # backtrack to the last known place we saw
        # this square from..
        while maze(loc) == 0:
            for val in DIRECTIONS.values():
                new_loc = loc + val
                if np.any(new_loc >= maze.dim):
                    continue
                new_g = self.g_values(new_loc)
                if g_val == (new_g + 1):
                    loc = new_loc
                    g_val = new_g
                    break

        return loc

    def get_best_path(self, point_b):
        return path_from_g_values(self.robot.map.data(), self.g_values.data(), 
                                  [0, 0], point_b)

    def reverse_heuristic(self):
        self.reverse = True


class _AStarState(object):
    """Helper object to wrap a location and its corresponding
    h, g, and f values for A*"""

    def __init__(self, direction, distance, location, h_value, g_value):
        self.direction = direction
        self.distance = distance
        self._loc = location
        self.h_value = h_value
        self.g_value = g_value

    @property
    def f_value(self):
        return float(self.g_value + self.h_value)

    @property
    def location(self):
        return np.array(self._loc)

    def __repr__(self):
        return '<State: {}, {}, {}>'.format(
            self.direction, self.location, self.f_value)

