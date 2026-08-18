"""AnalyticBilliard as a shared library.

    from hpcBilliards import compileScene, updateScene

    compiled = compileScene(SCENE, backend="openmp")
    rec = compiled.recordWithIterations(x, y, vx, vy, iterations=1000)
    ser = compiled.recordWithTime(x, y, vx, vy, tf=100.0, dt=0.01)

Two ways to record the same run: recordWithIterations gives one row per event,
recordWithTime one row per tick of a uniform grid.

A scene may also declare `basinObjects`, surfaces that END a run instead of
reflecting it. getBasins then keeps only where each particle got out -- one row
per particle and no trajectory at all -- giving up either at a time or after a
number of events:

    esc = getBasins(compiled, x, y, tf=100.0)        # give up at t = 100
    esc = getBasins(compiled, x, y, iterations=1000) # give up after 1000 events
    esc.id                                      # which basin, 0 = never escaped

Declare the objects once and compile; after that, parameters, the portal flips,
the basin positions and labels, gravity and deadTime all change without touching
the compiler. Only adding, removing or retyping an object needs a rebuild.
"""

from .binding import (CompiledScene, Escape, Record, compileScene, getBasins,
                      updateScene)
from .scene import SceneError

__all__ = ["compileScene", "updateScene", "getBasins", "CompiledScene",
           "Record", "Escape", "SceneError"]
