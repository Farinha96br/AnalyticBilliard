"""Ready-made tables. Every one is also here at the top level:

    from analyticBilliards import commonTables as T
    T.stadium(1.0, 1.0), T.nestedEllipses(10, g=1.0)

    classic      circle, oval, stadium, lemon, mushroom, poliLine, poliLineSeg,
                 flower, PolyCircle
    benchmarks   rectangle, nestedEllipses, nestedEllipseArcs, stackedLineSegs,
                 stackedLines, portalBox, lorentzGas
"""

from .benchmarks import (lorentzGas, nestedEllipseArcs, nestedEllipses,
                         portalBox, rectangle, stackedLines, stackedLineSegs)
from .classic import (PolyCircle, circle, flower, lemon, mushroom, oval,
                      poliLine, poliLineSeg, stadium)
