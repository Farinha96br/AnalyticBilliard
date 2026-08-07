// a small scene whose only purpose is to show what a portal does.
//
//   g++ -O2 -ffp-contract=off -o portalsegment.out PortalSegment.cpp
//   ./portalsegment.out > portalsegment.dat
//   python3 plotBox.py portalsegment.dat --out portalsegment.png --bounces 8 --g 1.0
//
// one portal stands vertical and the other lies flat, a quarter turn apart, which
// is where the interesting part shows: the velocity is carried across in the
// portal's own frame, so a particle arriving level leaves falling. the pair is
// the same size -- asPortal insists on it -- so the map is a rigid motion and a
// particle entering at one end comes out at the matching end.
//
// crossing does change a particle's height, and so its potential energy. that is
// fine for a picture; keep a pair level when the energy has to be conserved, as
// the pair in ComplexExample.cpp is.
#include "physics.h"
#include <cstdio>

int main() {
    const int Niter = 14; // few enough that a single path can be followed by eye
    const int Npart = 1;

    Object upright = makeLineSegment(-8.0, 2.0, -8.0, 8.0);
    Object flat    = makeLineSegment(2.0, 5.0, 8.0, 5.0);

    const int nObj = 6;
    Object objs[nObj] = {
        makeLine(0.0, 1.0, 0.0),   // floor    y = 0
        makeLine(0.0, 1.0, -10.0), // ceiling  y = 10
        makeLine(1.0, 0.0, 20.0),  // wall     x = -20
        makeLine(1.0, 0.0, -20.0), // wall     x = 20

        // one upright, one flat, so a crossing turns the particle by a quarter
        // turn. both are 6 long, as they have to be, and each names the other
        asPortal(upright, flat, -1.0),
        asPortal(flat, upright, -1.0),
    };

    // one particle, launched straight at the upright portal with nothing in the
    // way, so the quarter turn is the first thing that happens to it. a second
    // path over the top would only hide the one worth following
    state states[Npart] = {
        {-15.0, 5.0, 4.0, 2.0},
    };

    const int rowsPer = 2 * Niter + 1; // a portal event costs two rows
    Row *rows = new Row[Npart * rowsPer];
    int counts[Npart];

    for (int i = 0; i < Npart; ++i) {
        counts[i] = getCollisionStates(states[i], objs, nObj, Niter,
                                       &rows[i * rowsPer], rowsPer);
    }

    for (int k = 0; k < nObj; ++k) {
        if (objs[k].type == LINE) {
            printf("# line %.17g %.17g %.17g\n", objs[k].p[0], objs[k].p[1], objs[k].p[2]);
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

    // a teleport is the one place two rows share an instant: report them so the
    // numbers can be checked against the picture
    fprintf(stderr, "particle  rows  teleports\n");
    for (int i = 0; i < Npart; ++i) {
        int jumps = 0;
        for (int r = 1; r < counts[i]; ++r) {
            Row a = rows[i * rowsPer + r - 1], b = rows[i * rowsPer + r];
            if (a.t == b.t && (a.x != b.x || a.y != b.y)) ++jumps;
        }
        fprintf(stderr, "%8d %5d %10d\n", i, counts[i], jumps);
    }

    delete[] rows;
    return 0;
}
