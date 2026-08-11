import sympy as sp
from hpcBilliards import *



# build a test scene:
# A vertical Line a x = 0, b y = 1, c = 0 
# A Line legment from (10,0) to (10,7)
# A Elispe arc with center (0,2), and radius 1, forming a cup
# A Elipse at 


 
SCENE = {
    "g": 1.0,  # gravity
    "solidObjects": [
        # line: a*x + b*y + c = 0
        {"type": "line", "params": [0.0, 1.0, 0.0]},
        #line segment: (x1,y1) to (x2,y2)
        {"type": "lineSegment", "params": [10.0, 0.0, 10.0, 7.0]},
        # elipse arc: (x0,y0) center, a,b radii, and rotation angle theta
        {"type": "elipse", "params": [0.0, 1.0, 1.0, 1.0, 0.0]},
        # elipse arc: (x0,y0) center, a,b radii, and rotation angle theta stard and end angle
        {"type": "elipseArc", "params": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, -3.14]},
    ],
    "portalObjecs": [
        {"entry": "lineSegment", "params": [0.0, 1.0, 0.0, 1.0],
         "exit": "lineSegment", "params": [1.0, 0.0, 1.0, 0.0], 
         "tangentFlip": +1, "normalFlip": +1}
    ]

}

# TODO /\ PENSAR COMO ESCREVER ISSO SE

x = np.array([0.5, 0.5])  # initial position
y = np.array([1.0, 0.0])  # initial velocity
vx = np.array([0.0, 0.0])  # initial velocity x
vy = np.array([0.0, 0.0])  # initial velocity y



# This usage example run the scene for 1000 iterations and record the bounce and portal events.
x_rec, y_rec, vx_rec, vy_rec, eventType = recordScene(x,y,vx,vy,SCENE,iterations=1000,record=["bounce","portal"])



# This usage example runs the scene by time for 1000 seconds and record the bounce energy at the bounce events only. 
E_rec = recordScene(x,y,vx,vy,SCENE,tf=1000,record=["bounce"])

