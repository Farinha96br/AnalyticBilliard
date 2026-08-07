// both signs reversed: the velocity is turned right around.
//
//   g++ -O2 -ffp-contract=off -o portalflipboth.out PortalFlipBoth.cpp
//   ./portalflipboth.out > portalflipboth.dat
//   python3 plotBox.py portalflipboth.dat --out portalflipboth.png --bounces 12 --g 1.0
//
// the two signs are independent, and this is simply both at once. the exit
// point is still the one PortalPlain gives; only the direction differs.
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
                         asPortal(left, right, -1.0, -1.0),
                         asPortal(right, left, -1.0, -1.0));

    runScene("flipN -1  flipT -1   both reversed", objs, nObj, portalStart, 1, 25);
    return 0;
}
