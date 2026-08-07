// flipTangent = -1: the velocity ALONG the surface is reversed.
//
//   g++ -O2 -ffp-contract=off -o portalfliptangent.out PortalFlipTangent.cpp
//   ./portalfliptangent.out > portalfliptangent.dat
//   python3 plotBox.py portalfliptangent.dat --out portalfliptangent.png --bounces 12 --g 1.0
//
// the particle still comes out at the height it went in at -- the signs never
// move it -- but it now runs the other way along the exit. climbing in, it
// leaves descending. compare with PortalPlain: same dots, mirrored arrows.
//
// the box, the portal positions and the opening arc are shared with the other
// portal cases, so anything that differs between two of them is the pair
// itself. gravity is on, hence the arcs.
#include "sceneRun.h"

int main() {
    Object left = makeLineSegment(-8.0, 2.0, -8.0, 8.0);
    Object right = makeLineSegment(8.0, 2.0, 8.0, 8.0);

    Object objs[6];
    int nObj = portalBox(objs,
                         asPortal(left, right, 1.0, -1.0),
                         asPortal(right, left, 1.0, -1.0));

    runScene("flipN +1  flipT -1   vertical velocity reversed", objs, nObj, portalStart, 1, 25);
    return 0;
}
