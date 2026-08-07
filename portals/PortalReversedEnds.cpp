// the other control: no sign changed, the partner written backwards.
//
//   g++ -O2 -ffp-contract=off -I.. -o portalreversedends.out PortalReversedEnds.cpp
//   ./portalreversedends.out > portalreversedends.dat          (./checkHashPortal.sh does all six)
//   python3 ../plotBox.py portalreversedends.dat --out portalreversedends.png --bounces 12 --g 1.0
// flipNormal = -1 as well to keep the original side.
#include "physics.h" // pulls in geometry.h and types.h
#include <cstdio>

int main() {

    const int Niter = 1e6; 
    const int Npart = 1;

    Object left  = makeLineSegment(-8.0, 2.0, -8.0, 8.0);
    Object right = makeLineSegment( 8.0, 8.0,  8.0, 2.0);

    const int nObj = 6;
    Object objs[nObj] = {
        makeLine(0.0, 1.0, 0.0),   // floor    y = 0
        makeLine(0.0, 1.0, -10.0), // ceiling  y = 10
        makeLine(1.0, 0.0, 20.0),  // wall     x = -20
        makeLine(1.0, 0.0, -20.0), // wall     x = 20

        // flipNormal +1, flipTangent +1: PortalPlain's signs, untouched
        asPortal(left, right, 1.0, 1.0),
        asPortal(right, left, 1.0, 1.0),
    };

    // PortalPlain.cpp says why this start
    state states[Npart] = {
        {-16.0, 1.0, 5.75, 4.25},
    };

    const int rowsPer = 2 * Niter + 1;
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


    fprintf(stderr, "flipN +1  flipT +1, partner reversed   height swapped\n");
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
