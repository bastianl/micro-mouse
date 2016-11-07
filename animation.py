import pygame
from pygame import draw


class App(object):
 
    windowWidth = 800
    windowHeight = 600
 
    def __init__(self, maze, robot):
        self.maze = maze
        self.robot = robot
        pygame.init()
        self._surface = pygame.display.set_mode(
            (self.windowWidth, self.windowHeight))
        pygame.display.set_caption('Micro Mouse')

    def draw(self):
        self._surface.fill(0x000000)
        self.draw_maze()
        pygame.display.flip()
        self.draw_robot()
 
    def draw_maze(self):
        sq_size = 20
        w = 2 # line-width
        color = 0xffffff
        origin = 0
        for x in range(self.maze.dim):
            for y in range(self.maze.dim):
                if not self.maze.is_permissible([x,y], 'up'):
                    x_, y_ = origin + sq_size * x, origin + sq_size * (y+1)
                    draw.line(self._surface, color, (x_, y_), (x_, y_ + sq_size), w)
                if not self.maze.is_permissible([x,y], 'right'):
                    x_, y_ = origin + sq_size * (x+1), origin + sq_size * y
                    draw.line(self._surface, color, (x_, y_), (x_ + sq_size, y_), w)
                # if y == 0 and not testmaze.is_permissible([x,y], 'down'):
                # if x == 0 and not testmaze.is_permissible([x,y], 'left'):
        import ipdb; ipdb.set_trace()

    def place_robot(self):
        import ipdb; ipdb.set_trace()

    def on_cleanup(self):
        pygame.quit()
