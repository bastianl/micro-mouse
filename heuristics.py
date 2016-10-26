from random import random

from utils import get_updated_location

class BaseHeuristic(object):
    """Base class for heuristic objects. Stores whole robot
    action so we can the many properties on the robot to determine
    the best move."""

    def __init__(self, robot):
        self.robot = robot

    def get_move(self, current_state, open_states):
        raise NotImplemented()


class RandomHeuristic(BaseHeuristic):
    """Random heuristic. Make decisions randomly, but with a
    weighting factor to determine preference towards squares
    that have not been explored."""

    def __init__(self, robot, weight=5):
        self.robot = robot
        self.weight = weight

    def get_move(self, open_states):
        robot = self.robot
        for st in open_states:
            loc = get_updated_location(st[0], robot.location)
            try:
                heuristic = self.random_heuristic(loc, len(open_states))
            except IndexError as e:
                import ipdb; ipdb.set_trace()
            st.append(heuristic)

        # random heuristic uses the largest value
        direction, heuristic = sorted(open_states, key=lambda x: x[1],
                                      reverse=True)[0]
        print("Best heuristic out of {} choices: {}".format(
            len(open_states), heuristic))
        return direction

    def random_heuristic(self, loc, num_states):
        """Randomly provide a heuristic for a node.
        Weight is the multiplier for preference if that node
        has not yet been explored."""
        # prefer unexplored states
        maze = self.robot._map
        mult = self.weight * (maze[loc[0], loc[1]] == 0) + 1
        return random() / num_states * (mult)


def a_star_heuristic(loc, dim):
    """Get an optimistic prediction to the goal"""
    [x, y] = loc
    # to account for the 2x2 goal in the center
    x_dim = dim // 2 - 1 if x < dim // 2 else dim //2
    y_dim = dim // 2 - 1 if y < dim // 2 else dim //2
    return abs(x - x_dim) + abs(y - y_dim)