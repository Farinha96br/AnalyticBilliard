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


# initial contitions are a lists or numpy arrays

x = np.array([0.5, 0.5])  # initial position
y = np.array([1.0, 0.0])  # initial velocity
vx = np.array([0.0, 0.0])  # initial velocity x
vy = np.array([0.0, 0.0])  # initial velocity y



# compile the scene for the first time
# the backend can be "linear","openMP","gpu_openMP"
compiledScene = compileScene(SCENE, backend="openmp")

# This usage example runs the scene for 1000 events and records the bounce and
# portal ones. There is no "id" column: the particle index is the first axis of
# every array, so x_rec[i] is particle i.
x_rec, y_rec, vx_rec, vy_rec, evType = compiledScene.recordWithIterations(
    x, y, vx, vy, iterations=1000,
    eventType=["bounce", "portal"],
    save=["x", "y", "vx", "vy", "evType"])


# a change in the scene can be introduced
SCENE_2 = {
    "g": 0.5,  # gravity
    "solidObjects": [
        # line: a*x + b*y + c = 0
        {"type": "line", "params": [0.1, 1.0, 0.0]},
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

# the scene valuesa are updated
updateScene(SCENE_2, compiledScene)

# The energy at the bounce events only. save= decides what is allocated, not
# just what comes back, so this holds one column instead of six.
E_rec = compiledScene.recordWithIterations(
    x, y, vx, vy, iterations=1000,
    eventType=["bounce"], save=["Energy"]).energy

# The same run seen as a time series instead: one row every 0.01 up to t=1000,
# on a grid shared by every particle. Rows here are samples of a flight rather
# than events, so there is no eventType filter -- that is the other function.
series = compiledScene.recordWithTime(x, y, vx, vy, tf=1000.0, dt=0.01)

