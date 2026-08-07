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

        // flipNormal +1, flipTangent +1: the velocity crosses untouched
        asPortal(left, right, 1.0, 1.0),
        asPortal(right, left, 1.0, 1.0),
    };

    // one particle, not a fan of them. these scenes exist to show what a single
    // crossing does, and a second path laid over the first hides exactly the
    // part worth looking at.
    //
    // three things had to hold at once, and most starts fail at least one:
    //   - it flies clean into the left portal, touching nothing on the way, so
    //     the crossing that separates the cases is the first thing that happens
    //   - it crosses at y ~ 5.9, mid-segment, where a crossing cannot be
    //     misread as a graze past an endpoint
    //   - it carries real vertical speed across (vy ~ 2.9), or flipTangent
    //     would be reversing something too small to see
    // and none of the five cases may fall into a closed orbit. flipTangent is
    // the fussy one: reversing vy on a level pair is close to a time reversal,
    // so plenty of starts come back on themselves and retrace a single loop for
    // the whole run, drawing every arc on top of every other and showing nothing
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
    fprintf(stderr, "flipN +1  flipT +1   straight through\n");
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
