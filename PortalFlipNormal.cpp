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
// the box, the portal positions and the opening arc are the same in every
// portal case, so whatever differs between two of the pictures is the pair
// itself. gravity is on, hence the arcs.
#include "physics.h" // pulls in geometry.h and types.h
#include <cstdio>

int main() {

    const int Niter = 25; // few enough that a single path can be followed by eye
    const int Npart = 1;

    // the pair, named so each can point at the other
    Object left  = makeLineSegment(-8.0, 2.0, -8.0, 8.0);
    Object right = makeLineSegment( 8.0, 2.0,  8.0, 8.0);

    const int nObj = 6;
    Object objs[nObj] = {
        makeLine(0.0, 1.0, 0.0),   // floor    y = 0
        makeLine(0.0, 1.0, -10.0), // ceiling  y = 10
        makeLine(1.0, 0.0, 20.0),  // wall     x = -20
        makeLine(1.0, 0.0, -20.0), // wall     x = 20

        // flipNormal -1, flipTangent +1: the normal velocity is reversed
        asPortal(left, right, -1.0, 1.0),
        asPortal(right, left, -1.0, 1.0),
    };

    // the same opening arc as every other case; PortalPlain.cpp says why this
    // one and not another
    state states[Npart] = {
        {-16.0, 1.0, 5.75, 4.25},
    };

    // a portal event costs two rows, the arrival and the departure, so the
    // slice is sized for the worst case of every event being one
    const int rowsPer = 2 * Niter + 1;
    Row *rows = new Row[Npart * rowsPer];
    int counts[Npart];

    for (int i = 0; i < Npart; ++i) {
        counts[i] = getCollisionStates(states[i], objs, nObj, Niter,
                                       &rows[i * rowsPer], rowsPer);
    }

    // the scene, then one row per collision. a portal's surface is described
    // like any other object; the second record carries where it leads, which is
    // what lets the plot draw the link and the direction arrows
    for (int k = 0; k < nObj; ++k) {
        if (objs[k].type == LINE) {
            printf("# line %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2]);
        } else {
            printf("# segment %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3]);
        }
        if (objs[k].response == PORTAL) {
            printf("# portal %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                   objs[k].q[0], objs[k].q[1], objs[k].q[2], objs[k].q[3],
                   objs[k].q[4], objs[k].q[5], objs[k].q[6], objs[k].q[7],
                   objs[k].q[8], objs[k].q[9]);
        }
    }

    printf("ID t x y vx vy\n");
    for (int i = 0; i < Npart; ++i) {
        for (int r = 0; r < counts[i]; ++r) {
            Row w = rows[i * rowsPer + r];
            printf("%d %.17g %.17g %.17g %.17g %.17g\n", i, w.t, w.x, w.y, w.vx, w.vy);
        }
    }

    // a teleport is the one place two rows share an instant, so it is easy to
    // pick out. the first one each particle takes is the whole story, and its
    // dE is the check: the signs move the velocity and never the position, so
    // a level pair must come through at no cost at all
    fprintf(stderr, "flipN -1  flipT +1   turns back\n");
    for (int i = 0; i < Npart; ++i) {
        for (int r = 1; r < counts[i]; ++r) {
            Row a = rows[i * rowsPer + r - 1], b = rows[i * rowsPer + r];
            if (a.t != b.t || (a.x == b.x && a.y == b.y)) continue;
            double ea = 0.5 * (a.vx * a.vx + a.vy * a.vy) + g * a.y;
            double eb = 0.5 * (b.vx * b.vx + b.vy * b.vy) + g * b.y;
            fprintf(stderr, "  p%d  (%6.2f,%5.2f) v=(%5.2f,%5.2f) -> "
                            "(%6.2f,%5.2f) v=(%5.2f,%5.2f)   dE %.1e\n",
                    i, a.x, a.y, a.vx, a.vy, b.x, b.y, b.vx, b.vy, fabs(eb - ea));
            break;
        }
    }

    delete[] rows;

    return 0;
}
