/* Exact critical polynomial on a small torus by full enumeration.
 *
 * For every subset of occupied sites (site mode) or bonds (bond mode) of the
 * Lx x Ly torus of a cell graph, decide: no wrapping cluster / some winding /
 * cross cluster (two non-parallel windings), with the same union-find with
 * displacements as nz.c.  Prints, for every number m of occupied objects, the
 * counts of configurations with no wrap and with a cross:
 *     P_nowrap(p) = sum_m A_m p^m (1-p)^(K-m),  P_cross(p) = sum_m C_m p^m (1-p)^(K-m),
 * so the critical polynomial is P_cross - P_nowrap (Jacobsen-Scullard, q = 1).
 *
 * usage: exact CELLFILE site|bond Lx Ly        (K = number of objects <= 30)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int par[64], ddx[64], ddy[64], wnx[64], wny[64];

static int find(int i, int *tx, int *ty) {
    int sx = 0, sy = 0;
    while (par[i] >= 0) { sx += ddx[i]; sy += ddy[i]; i = par[i]; }
    *tx = sx; *ty = sy;
    return i;
}

/* returns 2 if a cross appears, 1 if a (new) winding, 0 otherwise */
static int join(int s, int t, int ex, int ey) {
    int ax, ay, bx, by, r1 = find(s, &ax, &ay), r2 = find(t, &bx, &by);
    int wx = ax - ex - bx, wy = ay - ey - by;
    if (r1 == r2) {
        if (!wx && !wy) return 0;
        if (!wnx[r1] && !wny[r1]) { wnx[r1] = wx; wny[r1] = wy; return 1; }
        return ((long)wnx[r1] * wy - (long)wny[r1] * wx) != 0 ? 2 : 1;
    }
    int big = r1, small = r2;
    if (par[r1] <= par[r2]) { par[r1] += par[r2]; par[r2] = r1; ddx[r2] = wx; ddy[r2] = wy; }
    else { big = r2; small = r1; par[r2] += par[r1]; par[r1] = r2; ddx[r1] = -wx; ddy[r1] = -wy; }
    if (wnx[small] || wny[small]) {
        if (!wnx[big] && !wny[big]) { wnx[big] = wnx[small]; wny[big] = wny[small]; return 1; }
        return ((long)wnx[big] * wny[small] - (long)wny[big] * wnx[small]) != 0 ? 2 : 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: exact CELLFILE site|bond Lx Ly\n"); return 1; }
    FILE *f = fopen(argv[1], "r");
    int nv, ne;
    if (!f || fscanf(f, "%d %d", &nv, &ne) != 2) return 1;
    int eu[256], ew[256], ea[256], eb[256];
    for (int k = 0; k < ne; k++) if (fscanf(f, "%d %d %d %d", &eu[k], &ew[k], &ea[k], &eb[k]) != 4) return 1;
    fclose(f);
    int bond = !strcmp(argv[2], "bond"), Lx = atoi(argv[3]), Ly = atoi(argv[4]);
    int N = Lx * Ly * nv, E = 0;
    static int Es[1024], Et[1024], Ex[1024], Ey[1024];
    for (int b = 0; b < Ly; b++)
        for (int a = 0; a < Lx; a++)
            for (int k = 0; k < ne; k++) {
                int a2 = ((a + ea[k]) % Lx + Lx) % Lx, b2 = ((b + eb[k]) % Ly + Ly) % Ly;
                Es[E] = (b * Lx + a) * nv + eu[k]; Et[E] = (b2 * Lx + a2) * nv + ew[k];
                Ex[E] = ea[k]; Ey[E] = eb[k]; E++;
            }
    int K = bond ? E : N;
    if (K > 30 || N > 64) { fprintf(stderr, "too large: K=%d N=%d\n", K, N); return 1; }
    long long A[32] = {0}, C[32] = {0}, W[32] = {0};
    for (uint64_t mask = 0; mask < (1ULL << K); mask++) {
        int m = __builtin_popcountll(mask), cross = 0, wrap = 0;
        for (int i = 0; i < N; i++) { par[i] = -1; ddx[i] = ddy[i] = wnx[i] = wny[i] = 0; }
        for (int e = 0; e < E && !cross; e++) {
            int ok = bond ? (int)((mask >> e) & 1) : (int)(((mask >> Es[e]) & 1) && ((mask >> Et[e]) & 1));
            if (!ok) continue;
            int r = join(Es[e], Et[e], Ex[e], Ey[e]);
            if (r) wrap = 1;
            if (r == 2) cross = 1;
        }
        if (!wrap) A[m]++;
        if (cross) C[m]++;
        if (wrap && !cross) W[m]++;
    }
    printf("# %s %s %dx%d K=%d\n# m nowrap cross other\n", argv[1], argv[2], Lx, Ly, K);
    for (int m = 0; m <= K; m++) printf("%d %lld %lld %lld\n", m, A[m], C[m], W[m]);
    return 0;
}
