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
    "collision": [
        # line: a*x + b*y + c = 0
        {"type": "line", "params": [0.0, 1.0, 0.0]},
        #line segment: (x1,y1) to (x2,y2)
        {"type": "lineSegment", "params": [10.0, 0.0, 10.0, 7.0]},
        # elipse arc: (x0,y0) center, a,b radii, and rotation angle theta
        {"type": "elipse", "params": [0.0, 1.0, 1.0, 1.0, 0.0]},
        # elipse arc: (x0,y0) center, a,b radii, and rotation angle theta stard and end angle
        {"type": "elipseArc", "params": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, -3.14]},
    ]
}



# final usage example:

# Get the final positions of a given scne

