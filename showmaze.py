from maze import Maze
import turtle
import sys

class MazeVis(object):
    sq_size = 20
    mapping = True

    def __init__(self, maze):
        window = turtle.Screen()
        wally = turtle.Turtle()
        wally.speed(0)
        wally.shape("circle")
        wally.shapesize(0.5, 0.5, 1)
        wally.penup()
        self.window = window
        self.wally = wally
        self.maze = maze

        # maze centered on (0,0), squares are 20 units in length.
        self.origin = maze.dim * self.sq_size / -2

    def draw_maze(self):
        maze = self.maze
        wally = self.wally
        origin = self.origin
        sq_size = self.sq_size
        wally.tracer(0, 0)
        wally.hideturtle()

        # iterate through squares one by one to decide where to draw walls
        for x in range(maze.dim):
            for y in range(maze.dim):
                if not maze.is_permissible([x,y], 'up'):
                    wally.goto(origin + sq_size * x, origin + sq_size * (y+1))
                    wally.setheading(0)
                    wally.pendown()
                    wally.forward(sq_size)
                    wally.penup()

                if not maze.is_permissible([x,y], 'right'):
                    wally.goto(origin + sq_size * (x+1), origin + sq_size * y)
                    wally.setheading(90)
                    wally.pendown()
                    wally.forward(sq_size)
                    wally.penup()

                # only check bottom wall if on lowest row
                if y == 0 and not maze.is_permissible([x,y], 'down'):
                    wally.goto(origin + sq_size * x, origin)
                    wally.setheading(0)
                    wally.pendown()
                    wally.forward(sq_size)
                    wally.penup()

                # only check left wall if on leftmost column
                if x == 0 and not maze.is_permissible([x,y], 'left'):
                    wally.goto(origin, origin + sq_size * y)
                    wally.setheading(90)
                    wally.pendown()
                    wally.forward(sq_size)
                    wally.penup()
        turtle.update()


    def draw_robot(self, location):
        # self.wally.clearstamps()
        origin = self.origin
        sq_size = self.sq_size
        [x, y] = location
        self.wally.goto(origin + sq_size * x + 0.5 * sq_size, 
                        origin + sq_size * y + 0.5 * sq_size)
        if not self.mapping:
            self.wally.color('red')
            self.wally.showturtle()
            self.wally.pendown()
        else:
            self.wally.stamp()
        turtle.update()

    def end(self):
        self.window.exitonclick()


if __name__ == '__main__':
    '''
    This function uses Python's turtle library to draw a picture of the maze
    given as an argument when running the script.
    '''

    # Create a maze based on input argument on command line.
    testmaze = Maze( str(sys.argv[1]) )

    # Intialize the window and drawing turtle.
    viz = MazeVis(testmaze)
    viz.draw_maze()

    viz.end()