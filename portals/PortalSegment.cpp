// a small scene whose only purpose is to show what a portal does.
//
//   g++ -O2 -ffp-contract=off -I.. -o portalsegment.out PortalSegment.cpp
//   ./portalsegment.out > portalsegment.dat          (./checkHashPortal.sh does all six)
//   python3 ../plotBox.py portalsegment.dat --out portalsegment.png --bounces 8 --g 1.0

#include "physics.h"
#include <cstdio>

int main() {
    const int Niter = 1e6; 
    const int Npart = 1;

    Object upright = makeLineSegment(-8.0, 2.0, -8.0, 8.0);
    Object flat    = makeLineSegment(2.0, 5.0, 8.0, 5.0);

    const int nObj = 6;
    Object objs[nObj] = {
        makeLine(0.0, 1.0, 0.0),   // floor    y = 0
        makeLine(0.0, 1.0, -10.0), // ceiling  y = 10
        makeLine(1.0, 0.0, 20.0),  // wall     x = -20
        makeLine(1.0, 0.0, -20.0), // wall     x = 20

        asPortal(upright, flat, -1.0),
        asPortal(flat, upright, -1.0),
    };

    state states[Npart] = {
        {-15.0, 5.0, 4.0, 2.0},
    };

    const int rowsPer = 2 * Niter + 1; // a portal event costs two rows
    Row *rows = new Row[Npart * rowsPer];
    int counts[Npart];

#ifdef _OPENMP
#pragma omp target teams distribute parallel for \
    map(to : states[0 : Npart], objs[0 : nObj])  \
    map(from : rows[0 : Npart * rowsPer], counts[0 : Npart])
#endif
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
