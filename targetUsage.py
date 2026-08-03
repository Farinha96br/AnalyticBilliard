import sympy as sp



# this is an example of target usage of the library

# defin the root symbols (x,y) coordinates
x, y = sp.symbols('x y')

# Scene object types




# build a test scene:
# A vertical Line a x = 0, b y = 1, c = 0 
# A Line legment from (10,0) to (10,7)
# A Elispe arc with center (0,2), and radius 1, forming a cup
# A Elipse at 


 
Scene = {
    "g": 1.0,  # gravity
    "objs": [
        # line: a*x + b*y + c = 0
        {"type": "line", "a": 0.0, "b": 1.0, "c": 0.0},
        {"type": "lineSegment", "a": 0.0, "b": 1.0, "c": -2.0},
        {"type": "elipse", "x0": 0.0, "y0": 1.0, "a": 1.0, "b": 1.0, "theta": 0.0},
        {"type": "elipse", "x0": 0.0, "y0": 1.0, "a": 1.0, "b": 1.0, "theta": 0.0, "phi1": 0.0, "phi2": -3.14},

}



# final usage example:

# Get the final positions of a given scne

