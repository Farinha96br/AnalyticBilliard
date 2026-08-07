// flipNormal = -1: the velocity THROUGH the surface is reversed.
//
//   g++ -O2 -ffp-contract=off -o portalflipnormal.out PortalFlipNormal.cpp
//   ./portalflipnormal.out > portalflipnormal.dat
//   python3 plotBox.py portalflipnormal.dat --out portalflipnormal.png --bounces 12 --g 1.0
//
// instead of carrying on out the far side, the particle leaves by the face it
// arrived at, so it turns back into the space between the portals. the height
// is untouched, as with every sign.
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
                         asPortal(left, right, -1.0, 1.0),
                         asPortal(right, left, -1.0, 1.0));

    runScene("flipN -1  flipT +1   turns back", objs, nObj, portalStart, 1, 25);
    return 0;
}
