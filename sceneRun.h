#pragma once
/*
    the shared half of the small portal examples: run the particles, print the
    scene and the table, and report the first crossing of each.

    it lives here so that a case file holds nothing but its scene. reading one of
    them should show what that case does and nothing else -- the differences
    between them are a sign or an endpoint order, and those would be lost in
    fifty lines of identical printf.
*/
#include "physics.h"
#include <cstdio>

// one line per object: p[] means something different per type, so a single fixed
// format would misread every one. a portal adds a second record saying where it
// leads, which is what lets the plot draw the link and the direction arrows
inline void printScene(const Object *objs, int nObj) {
    for (int k = 0; k < nObj; ++k) {
        switch (objs[k].type) {
        case LINE:
            printf("# line %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2]);
            break;
        case LINE_SEGMENT:
            printf("# segment %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3]);
            break;
        case ELIPSE:
            printf("# elipse %.17g %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3], objs[k].p[4]);
            break;
        case ELIPSE_ARC:
            printf("# elipsearc %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                   objs[k].p[0], objs[k].p[1], objs[k].p[2], objs[k].p[3], objs[k].p[4],
                   objs[k].p[7], objs[k].p[8]);
            break;
        }
        if (objs[k].response == PORTAL) {
            const double *q = objs[k].q;
            printf("# portal %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                   q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]);
        }
    }
}

// a teleport is the one place two rows share an instant, so it is easy to pick
// out and worth reporting: the entry and exit of the first one each particle
// takes, and whether its energy came through untouched
inline void reportCrossings(const Row *rows, const int *counts, int nPart, int rowsPer) {
    for (int i = 0; i < nPart; ++i) {
        for (int r = 1; r < counts[i]; ++r) {
            Row a = rows[i * rowsPer + r - 1], b = rows[i * rowsPer + r];
            if (a.t != b.t || (a.x == b.x && a.y == b.y)) continue;
            double ea = 0.5 * (a.vx * a.vx + a.vy * a.vy) + g * a.y;
            double eb = 0.5 * (b.vx * b.vx + b.vy * b.vy) + g * b.y;
            fprintf(stderr, "  p%d  (%6.2f,%5.2f) v=(%5.2f,%5.2f) -> "
                            "(%6.2f,%5.2f) v=(%5.2f,%5.2f)   dE %.1e\n",
                    i, a.x, a.y, a.vx, a.vy, b.x, b.y, b.vx, b.vy, fabs(eb - ea));
            break; // the first crossing tells the whole story
        }
    }
}

inline void runScene(const char *what, const Object *objs, int nObj,
                     const state *init, int nPart, int nIter) {
    int rowsPer = 2 * nIter + 1; // a portal event costs two rows
    Row *rows = new Row[nPart * rowsPer];
    int *counts = new int[nPart];

    for (int i = 0; i < nPart; ++i) {
        counts[i] = getCollisionStates(init[i], objs, nObj, nIter,
                                       &rows[i * rowsPer], rowsPer);
    }

    printScene(objs, nObj);
    printf("ID t x y vx vy\n");
    for (int i = 0; i < nPart; ++i) {
        for (int r = 0; r < counts[i]; ++r) {
            Row w = rows[i * rowsPer + r];
            printf("%d %.17g %.17g %.17g %.17g %.17g\n", i, w.t, w.x, w.y, w.vx, w.vy);
        }
    }

    fprintf(stderr, "%s\n", what);
    reportCrossings(rows, counts, nPart, rowsPer);

    delete[] rows;
    delete[] counts;
}

// every portal case uses the same box and the same opening arc into the left
// portal, so that the only thing telling two of them apart is the portal pair
inline int portalBox(Object *objs, Object left, Object right) {
    int n = 0;
    objs[n++] = makeLine(0.0, 1.0, 0.0);   // floor    y = 0
    objs[n++] = makeLine(0.0, 1.0, -10.0); // ceiling  y = 10
    objs[n++] = makeLine(1.0, 0.0, 20.0);  // wall     x = -20
    objs[n++] = makeLine(1.0, 0.0, -20.0); // wall     x = 20
    objs[n++] = left;
    objs[n++] = right;
    return n;
}

// one particle, not a fan of them. these scenes exist to show what a single
// crossing does, and a second path laid over the first hides exactly the part
// worth looking at -- with one, every arc on the picture belongs to the story.
//
// three things had to hold at once, and most starts fail at least one:
//   - it flies clean into the left portal, touching nothing on the way, so the
//     crossing that separates the cases is the first thing that happens
//   - it crosses at y ~ 5.9, mid-segment, where a crossing cannot be misread as
//     a graze past an endpoint
//   - it carries real vertical speed across (vy ~ 2.9), or flipTangent would be
//     reversing something too small to see
// and none of the five may fall into a closed orbit. flipTangent is the fussy
// one: reversing vy on a level pair is close to a time reversal, so plenty of
// starts come back on themselves and retrace one loop for the whole run, which
// draws every arc on top of every other and shows nothing at all
static const state portalStart[1] = {
    {-16.0, 1.0, 5.75, 4.25},
};
