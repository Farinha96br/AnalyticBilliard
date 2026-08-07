// a portal pair glued the plain way: nothing is reversed.
//
//   g++ -O2 -ffp-contract=off -o portalplain.out PortalPlain.cpp
//   ./portalplain.out > portalplain.dat
//   python3 plotBox.py portalplain.dat --out portalplain.png --bounces 12 --g 1.0
//
// a particle crosses at the height it arrived at and carries on in the same
// direction. the other cases are this one with a sign or an ordering changed,
// so it is the thing to compare them against.
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
                         asPortal(left, right, 1.0, 1.0),
                         asPortal(right, left, 1.0, 1.0));

    runScene("flipN +1  flipT +1   straight through", objs, nObj, portalStart, 1, 25);
    return 0;
}
