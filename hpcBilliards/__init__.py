"""AnalyticBilliard as a shared library.

    from hpcBilliards import Scene, LineSegment, compileScene, updateScene

    scene = Scene(g=1.0, solidObjects=[LineSegment(0, 0, 1, 0), ...])
    compiled = compileScene(scene, backend="openmp")
    rec = compiled.recordWithIterations(x, y, vx, vy, iterations=1000)
    ser = compiled.recordWithTime(x, y, vx, vy, tf=100.0, dt=0.01)

Scenes are built from typed objects -- Line, LineSegment, Elipse, ElipseArc and
Portal -- so a missing parameter or a misspelled name is an error where you
wrote it, not a scene that compiles and quietly behaves wrong.

Two ways to record the same run: recordWithIterations gives one row per event,
recordWithTime one row per tick of a uniform grid.

A scene may also declare `basinObjects`, a dict of label -> shapes that END a
run instead of reflecting it. getBasins then keeps only where each particle got
out -- one row per particle and no trajectory at all -- giving up either at a
time or after a number of events:

    esc = getBasins(compiled, x, y, tf=100.0)        # give up at t = 100
    esc = getBasins(compiled, x, y, iterations=1000) # give up after 1000 events
    esc.id                                      # which basin, 0 = never escaped

Declare the objects once and compile; after that, parameters, the portal flips,
the basin positions and labels, gravity and deadTime all change without touching
the compiler. Only adding, removing or retyping an object needs a rebuild. A
Scene is mutable, so that is an edit and a push:

    scene.g = 0.3
    updateScene(scene, compiled)                # no compiler runs
"""

from .binding import (CompiledScene, Escape, Record, compileScene, getBasins,
                      updateScene)
from .scene import (Elipse, ElipseArc, Line, LineSegment, Portal, Scene,
                    SceneError)

__all__ = ["compileScene", "updateScene", "getBasins", "CompiledScene",
           "Record", "Escape", "SceneError", "Scene", "Line", "LineSegment",
           "Elipse", "ElipseArc", "Portal"]
