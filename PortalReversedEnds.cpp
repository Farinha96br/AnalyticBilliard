// the other control: no sign changed, the partner written backwards.
//
//   g++ -O2 -ffp-contract=off -o portalreversedends.out PortalReversedEnds.cpp
//   ./portalreversedends.out > portalreversedends.dat
//   python3 plotBox.py portalreversedends.dat --out portalreversedends.png --bounces 12 --g 1.0
//
// PortalPlain's signs exactly, and yet the heights swap: the entry is mirrored
// about the middle of the pair, so arriving at 5.95 it leaves at 4.05. WHERE a
// particle emerges is decided by the order the partner's endpoints were written
// in, never by the signs.
//
// this is also the only case whose energy changes, and for an honest reason:
// the crossing genuinely lifts or drops the particle. reversing the order also
// turns the exit face over, since the normal is the tangent turned a quarter
// turn -- pass flipNormal = -1 as well to keep the original side.
//
// the box, the portal positions and the opening arc are shared with the other
// portal cases, so anything that differs between two of them is the pair
// itself. gravity is on, hence the arcs.
#include "sceneRun.h"

int main() {
    Object left = makeLineSegment(-8.0, 2.0, -8.0, 8.0);
    // the same segment as every other case, written the other way round.
    // that ordering is the whole difference here: it is what decides which
    // end of the partner the entry maps onto
    Object right = makeLineSegment(8.0, 8.0, 8.0, 2.0);

    Object objs[6];
    int nObj = portalBox(objs,
                         asPortal(left, right, 1.0, 1.0),
                         asPortal(right, left, 1.0, 1.0));

    runScene("flipN +1  flipT +1, partner reversed   height swapped", objs, nObj, portalStart, 1, 25);
    return 0;
}
