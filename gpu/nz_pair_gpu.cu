// Coupled Newman-Ziff runs on the two members of a slip pair, many torus pairs at once (one pair per GPU thread).
// Same algorithm as gpu/nz_gpu.cu; coupling and torus geometry as in nz/nz_pair.c: the two cell files number
// their vertices and edges identically (analysis/slip_coupled_cells.py), each sample occupies the elements of both
// tori in one common random order, and the staggered member (second cell) has a twisted boundary: a cell
// (a, b + Ly) is identified with (a + TWIST, b), so that both members live on the same torus <Lx T1, Ly T2>.
// The threads are split into G groups (thread t -> group t mod G) with separate histograms; group g is written to
// OUTPREFIX__run<g>_E.txt and _S.txt (format of nz.c), so the error of the difference follows from the groups.
//
// usage: nz_pair_gpu CELL_E CELL_S site|bond Lx Ly TWIST SECONDS SEED OUTPREFIX GROUPS [THREADS]
// Builds with nvcc (GPU) or with a C++ compiler (serial CPU check).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

struct Node {
    int32_t par;
    int16_t dx, dy;
};

struct Lat {
    int nv, ne, Lx, Ly, mode, twist;  // mode 0 site, 1 bond
    int32_t N, Nobj;
    const int *eu, *ew, *ea, *eb;    // cell edges
    const int *aoff, *aw, *aa, *ab;  // per-orbit adjacency (CSR)
};

static const int32_t EMPTY = INT32_MIN;

struct Rng {  // xoshiro128**
    uint32_t s[4];
    HD static uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
    HD uint32_t next() {
        uint32_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 11);
        return r;
    }
    HD void seed(uint64_t x) {
        for (int i = 0; i < 4; i++) {  // splitmix64
            uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z ^= z >> 31;
            s[i] = (uint32_t)(z ^ (z >> 32));
        }
        if (!(s[0] | s[1] | s[2] | s[3])) s[0] = 1;
    }
    HD uint32_t bounded(uint32_t n) {  // Lemire, unbiased
        uint64_t m = (uint64_t)next() * n;
        uint32_t l = (uint32_t)m;
        if (l < n) {
            uint32_t t = (uint32_t)(-n) % n;
            while (l < t) { m = (uint64_t)next() * n; l = (uint32_t)m; }
        }
        return (uint32_t)(m >> 32);
    }
};

HD inline int32_t find(Node *nd, int32_t i, int32_t N, int &sx, int &sy, Node &R) {
    sx = 0; sy = 0;
    Node ni = nd[i];
    while (ni.par >= 0) {
        int32_t p = ni.par;
        if (p >= N) return -1;
        Node np = nd[p];
        if (np.par >= 0) {
            ni.par = np.par;
            ni.dx = (int16_t)(ni.dx + np.dx);
            ni.dy = (int16_t)(ni.dy + np.dy);
            nd[i] = ni;
            sx += ni.dx; sy += ni.dy;
            i = ni.par;
            ni = nd[i];
        } else {
            sx += ni.dx; sy += ni.dy;
            i = p;
            ni = np;
        }
    }
    R = ni;
    return i;
}

HD inline int cross_w(int ax, int ay, int bx, int by) { return (int64_t)ax * by - (int64_t)ay * bx != 0; }

HD inline int join(Node *nd, int32_t N, int32_t s, int32_t t, int ex, int ey) {
    int ax, ay, bx, by;
    Node Rs, Rt;
    if ((uint32_t)s >= (uint32_t)N || (uint32_t)t >= (uint32_t)N) return 8;
    int32_t r1 = find(nd, s, N, ax, ay, Rs);
    int32_t r2 = find(nd, t, N, bx, by, Rt);
    if (r1 < 0 || r2 < 0) return 8;
    int wx = ax - ex - bx, wy = ay - ey - by;
    if (r1 == r2) {
        if (wx == 0 && wy == 0) return 0;
        int f = (wx != 0) | ((wy != 0) << 1);
        if (Rs.dx == 0 && Rs.dy == 0) {
            Rs.dx = (int16_t)wx; Rs.dy = (int16_t)wy; nd[r1] = Rs;
        } else if (cross_w(Rs.dx, Rs.dy, wx, wy)) {
            f |= 4;
        }
        return f;
    }
    int f = 0;
    if (Rs.par <= Rt.par) {
        Rs.par += Rt.par;
        if (Rt.dx != 0 || Rt.dy != 0) {
            if (Rs.dx == 0 && Rs.dy == 0) { Rs.dx = Rt.dx; Rs.dy = Rt.dy; }
            else if (cross_w(Rs.dx, Rs.dy, Rt.dx, Rt.dy)) f = 4;
        }
        nd[r1] = Rs;
        Node c; c.par = r1; c.dx = (int16_t)wx; c.dy = (int16_t)wy;
        nd[r2] = c;
    } else {
        Rt.par += Rs.par;
        if (Rs.dx != 0 || Rs.dy != 0) {
            if (Rt.dx == 0 && Rt.dy == 0) { Rt.dx = Rs.dx; Rt.dy = Rs.dy; }
            else if (cross_w(Rt.dx, Rt.dy, Rs.dx, Rs.dy)) f = 4;
        }
        nd[r2] = Rt;
        Node c; c.par = r2; c.dx = (int16_t)(-wx); c.dy = (int16_t)(-wy);
        nd[r1] = c;
    }
    return f;
}

// target cell of (a, b) + (da, db) on the (possibly twisted) torus
HD inline int32_t cell_of(const Lat &L, int a, int b, int da, int db) {
    int b2 = b + db, jw = 0;
    if (b2 >= L.Ly) { b2 -= L.Ly; jw = 1; } else if (b2 < 0) { b2 += L.Ly; jw = -1; }
    int a2 = a + da + jw * L.twist;
    a2 = (a2 % L.Lx + L.Lx) % L.Lx;
    return b2 * L.Lx + a2;
}

// add element x to lattice L; returns winding flags (8: corrupt)
HD inline int add(const Lat &L, Node *nd, int32_t x) {
    int w = 0;
    if (L.mode) {
        int32_t c = x / L.ne, k = x - c * L.ne;
        int32_t a = c % L.Lx, b = c / L.Lx;
        int da = L.ea[k], db = L.eb[k];
        w = join(nd, L.N, c * L.nv + L.eu[k], cell_of(L, a, b, da, db) * L.nv + L.ew[k], da, db);
    } else {
        Node r; r.par = -1; r.dx = 0; r.dy = 0;
        nd[x] = r;
        int32_t c = x / L.nv, u = x - c * L.nv;
        int32_t a = c % L.Lx, b = c / L.Lx;
        for (int q = L.aoff[u]; q < L.aoff[u + 1]; q++) {
            int da = L.aa[q], db = L.ab[q];
            int32_t y = cell_of(L, a, b, da, db) * L.nv + L.aw[q];
            if ((uint32_t)y >= (uint32_t)L.N) { w |= 8; break; }
            if (nd[y].par != EMPTY) w |= join(nd, L.N, x, y, da, db);
        }
    }
    return w;
}

// one coupled sample; r[q] = {n_h, n_v, n_x} for member q (n_x = -2 on corruption)
HD inline void pair_sample(const Lat &A, const Lat &B, Node *na, Node *nb, int32_t *order, Rng &rng, int32_t r[2][3]) {
    Node init; init.dx = 0; init.dy = 0; init.par = A.mode ? -1 : EMPTY;
    for (int32_t i = 0; i < A.N; i++) { na[i] = init; nb[i] = init; }
    for (int q = 0; q < 2; q++) { r[q][0] = -1; r[q][1] = -1; r[q][2] = -1; }
    int done = 0;
    for (int32_t n = 0; n < A.Nobj && done < 2; n++) {
        uint32_t j = (uint32_t)n + rng.bounded((uint32_t)(A.Nobj - n));
        int32_t x = order[j]; order[j] = order[n]; order[n] = x;
        if ((uint32_t)x >= (uint32_t)A.Nobj) { r[0][2] = r[1][2] = -2; return; }
        for (int q = 0; q < 2; q++) {
            if (r[q][2] != -1) continue;
            int w = add(q ? B : A, q ? nb : na, x);
            if (w & 8) { r[0][2] = r[1][2] = -2; return; }
            if (w) {
                if ((w & 1) && r[q][0] < 0) r[q][0] = n + 1;
                if ((w & 2) && r[q][1] < 0) r[q][1] = n + 1;
                if (w & 4) { r[q][2] = n + 1; done++; }
            }
        }
    }
}

#ifdef __CUDACC__
typedef unsigned long long ull;

__global__ void kernel(Lat A, Lat B, Node *nodes, int32_t *orders, ull *hist, uint64_t seed, int launch, int samples,
                       int nthreads, int groups, int *bad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nthreads) return;
    Node *na = nodes + (size_t)t * 2 * A.N, *nb = na + A.N;
    int32_t *order = orders + (size_t)t * A.Nobj;
    Rng rng;
    rng.seed(seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1)) ^ (0xD1B54A32D192ED03ULL * (uint64_t)(launch + 1)));
    size_t H = (size_t)A.Nobj + 1;
    ull *hg = hist + (size_t)(t % groups) * 10 * H;
    for (int s = 0; s < samples; s++) {
        int32_t r[2][3];
        pair_sample(A, B, na, nb, order, rng, r);
        if (r[0][2] < 0 || r[1][2] < 0 || r[0][0] < 0 || r[0][1] < 0 || r[1][0] < 0 || r[1][1] < 0) {
            atomicAdd(bad, 1); continue;
        }
        for (int q = 0; q < 2; q++) {
            ull *h = hg + (size_t)q * 5 * H;
            int32_t nh = r[q][0], nvv = r[q][1];
            atomicAdd(&h[0 * H + nh], 1ULL);
            atomicAdd(&h[1 * H + nvv], 1ULL);
            atomicAdd(&h[2 * H + (nh < nvv ? nh : nvv)], 1ULL);
            atomicAdd(&h[3 * H + (nh > nvv ? nh : nvv)], 1ULL);
            atomicAdd(&h[4 * H + r[q][2]], 1ULL);
        }
    }
}

__global__ void init_orders(int32_t *orders, int32_t Nobj, int nthreads) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t tot = (size_t)Nobj * nthreads;
    for (; i < tot; i += (size_t)gridDim.x * blockDim.x) orders[i] = (int32_t)(i % Nobj);
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(3); } } while (0)

template <class T> static T *upload(const std::vector<T> &v) {
    T *d; CK(cudaMalloc(&d, (v.size() ? v.size() : 1) * sizeof(T)));
    if (v.size()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
#endif

struct Cell {
    int nv = 0, ne = 0, maxoff = 1;
    std::vector<int> eu, ew, ea, eb, aoff, aw, aa, ab;
    bool read(const char *fn) {
        FILE *f = fopen(fn, "r");
        if (!f) { perror(fn); return false; }
        if (fscanf(f, "%d %d", &nv, &ne) != 2) return false;
        eu.resize(ne); ew.resize(ne); ea.resize(ne); eb.resize(ne);
        for (int k = 0; k < ne; k++) {
            if (fscanf(f, "%d %d %d %d", &eu[k], &ew[k], &ea[k], &eb[k]) != 4) return false;
            maxoff = std::max(maxoff, std::max(abs(ea[k]), abs(eb[k])));
        }
        fclose(f);
        aoff.assign(nv + 1, 0); aw.resize(2 * ne); aa.resize(2 * ne); ab.resize(2 * ne);
        for (int k = 0; k < ne; k++) { aoff[eu[k] + 1]++; aoff[ew[k] + 1]++; }
        for (int u = 0; u < nv; u++) aoff[u + 1] += aoff[u];
        std::vector<int> fill(aoff.begin(), aoff.end() - 1);
        for (int k = 0; k < ne; k++) {
            int q = fill[eu[k]]++; aw[q] = ew[k]; aa[q] = ea[k]; ab[q] = eb[k];
            q = fill[ew[k]]++; aw[q] = eu[k]; aa[q] = -ea[k]; ab[q] = -eb[k];
        }
        return true;
    }
};

int main(int argc, char **argv) {
    if (argc < 11) {
        fprintf(stderr, "usage: nz_pair_gpu CELL_E CELL_S site|bond Lx Ly TWIST SECONDS SEED OUTPREFIX GROUPS [THREADS]\n");
        return 1;
    }
    Cell CE, CS;
    if (!CE.read(argv[1]) || !CS.read(argv[2])) return 1;
    if (CE.nv != CS.nv || CE.ne != CS.ne) { fprintf(stderr, "cells differ in size\n"); return 1; }
    Lat A, B;
    A.nv = B.nv = CE.nv; A.ne = B.ne = CE.ne;
    A.mode = B.mode = strcmp(argv[3], "bond") == 0;
    A.Lx = B.Lx = atoi(argv[4]); A.Ly = B.Ly = atoi(argv[5]);
    A.twist = 0; B.twist = atoi(argv[6]);
    A.N = B.N = A.Lx * A.Ly * A.nv;
    A.Nobj = B.Nobj = A.mode ? A.Lx * A.Ly * A.ne : A.N;
    double seconds = atof(argv[7]);
    uint64_t seed = strtoull(argv[8], NULL, 10);
    std::string prefix = argv[9];
    int groups = atoi(argv[10]);
    int maxoff = std::max(CE.maxoff, CS.maxoff);
    if ((int64_t)A.Lx * A.Ly * maxoff >= 32000 || A.Lx >= 16000 || A.Ly >= 16000 || (B.twist && A.Ly % 2)) {
        fprintf(stderr, "torus too large for int16 displacements, or odd Ly with twist\n");
        return 1;
    }
    size_t H = (size_t)A.Nobj + 1;
    std::vector<unsigned long long> hist((size_t)groups * 10 * H, 0);
    long long M = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    int nthreads = 0;
#ifdef __CUDACC__
    size_t per = 2 * (size_t)A.N * sizeof(Node) + (size_t)A.Nobj * sizeof(int32_t);
    const char *gm = getenv("GPUQ_GPU_MEM_GB");
    double budget = (gm ? atof(gm) : 4.0) - 1.5;
    if (budget < 0.5) budget = 0.5;
    nthreads = argc > 11 ? atoi(argv[11]) : 131072;
    size_t cap = (size_t)(budget * 1073741824.0 * 0.85) / per;
    if ((size_t)nthreads > cap) nthreads = (int)cap;
    nthreads = nthreads / 128 * 128;
    if (nthreads < 128) { fprintf(stderr, "not enough GPU memory\n"); return 1; }
    A.eu = upload(CE.eu); A.ew = upload(CE.ew); A.ea = upload(CE.ea); A.eb = upload(CE.eb);
    A.aoff = upload(CE.aoff); A.aw = upload(CE.aw); A.aa = upload(CE.aa); A.ab = upload(CE.ab);
    B.eu = upload(CS.eu); B.ew = upload(CS.ew); B.ea = upload(CS.ea); B.eb = upload(CS.eb);
    B.aoff = upload(CS.aoff); B.aw = upload(CS.aw); B.aa = upload(CS.aa); B.ab = upload(CS.ab);
    Node *nodes; int32_t *orders; ull *dh; int *dbad;
    CK(cudaMalloc(&nodes, per * nthreads));
    orders = (int32_t *)((char *)nodes + 2 * (size_t)A.N * sizeof(Node) * nthreads);
    CK(cudaMalloc(&dh, hist.size() * sizeof(ull)));
    CK(cudaMemset(dh, 0, hist.size() * sizeof(ull)));
    CK(cudaMalloc(&dbad, sizeof(int)));
    CK(cudaMemset(dbad, 0, sizeof(int)));
    init_orders<<<1024, 256>>>(orders, A.Nobj, nthreads);
    CK(cudaDeviceSynchronize());
    int samples = 1, launch = 0;
    while (true) {
        double s0 = elapsed();
        kernel<<<nthreads / 128, 128>>>(A, B, nodes, orders, dh, seed, launch++, samples, nthreads, groups, dbad);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        double tl = elapsed() - s0;
        M += (long long)samples * nthreads;
        double left = seconds - elapsed();
        if (left <= 0) break;
        double per_sample = tl / samples;
        int want = (int)(std::min(2.0, left) / per_sample);
        samples = std::max(1, std::min(want, samples * 4));
        if (per_sample * samples > left * 1.05 && samples > 1) samples = std::max(1, (int)(left / per_sample));
    }
    CK(cudaMemcpy(hist.data(), dh, hist.size() * sizeof(ull), cudaMemcpyDeviceToHost));
    int bad = 0;
    CK(cudaMemcpy(&bad, dbad, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad) { fprintf(stderr, "%d corrupt samples\n", bad); return 2; }
#else
    // serial CPU check of the same code path
    A.eu = CE.eu.data(); A.ew = CE.ew.data(); A.ea = CE.ea.data(); A.eb = CE.eb.data();
    A.aoff = CE.aoff.data(); A.aw = CE.aw.data(); A.aa = CE.aa.data(); A.ab = CE.ab.data();
    B.eu = CS.eu.data(); B.ew = CS.ew.data(); B.ea = CS.ea.data(); B.eb = CS.eb.data();
    B.aoff = CS.aoff.data(); B.aw = CS.aw.data(); B.aa = CS.aa.data(); B.ab = CS.ab.data();
    nthreads = argc > 11 ? atoi(argv[11]) : 4 * groups;
    std::vector<Node> nodes(2 * (size_t)A.N * nthreads);
    std::vector<int32_t> orders((size_t)A.Nobj * nthreads);
    for (size_t i = 0; i < orders.size(); i++) orders[i] = (int32_t)(i % A.Nobj);
    int launch = 0;
    while (elapsed() < seconds) {
        for (int t = 0; t < nthreads; t++) {
            Rng rng;
            rng.seed(seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1)) ^ (0xD1B54A32D192ED03ULL * (uint64_t)(launch + 1)));
            Node *na = &nodes[(size_t)t * 2 * A.N], *nb = na + A.N;
            unsigned long long *hg = &hist[(size_t)(t % groups) * 10 * H];
            for (int s = 0; s < 64; s++) {
                int32_t r[2][3];
                pair_sample(A, B, na, nb, &orders[(size_t)t * A.Nobj], rng, r);
                if (r[0][2] < 0 || r[1][2] < 0) { fprintf(stderr, "corrupt sample\n"); return 2; }
                for (int q = 0; q < 2; q++) {
                    unsigned long long *h = hg + (size_t)q * 5 * H;
                    int32_t nh = r[q][0], nvv = r[q][1];
                    h[0 * H + nh]++; h[1 * H + nvv]++;
                    h[2 * H + (nh < nvv ? nh : nvv)]++; h[3 * H + (nh > nvv ? nh : nvv)]++;
                    h[4 * H + r[q][2]]++;
                }
                M++;
            }
        }
        launch++;
    }
#endif
    double wall = elapsed();
    for (int g = 0; g < groups; g++) {
        for (int q = 0; q < 2; q++) {
            char fn[1024];
            snprintf(fn, sizeof fn, "%s__run%03d_%s.txt", prefix.c_str(), g, q ? "S" : "E");
            FILE *o = fopen(fn, "w");
            if (!o) { perror(fn); return 1; }
            const unsigned long long *h = &hist[(size_t)g * 10 * H + (size_t)q * 5 * H];
            long long Mg = 0;
            for (size_t k = 0; k < H; k++) Mg += (long long)h[4 * H + k];
            fprintf(o, "# cell %s mode %s Lx %d Ly %d nv %d ne %d N %lld samples %lld seed %s coupled twist %d group %d\n",
                    argv[1 + q], A.mode ? "bond" : "site", A.Lx, A.Ly, A.nv, A.ne, (long long)A.Nobj, Mg, argv[8],
                    q ? B.twist : 0, g);
            fprintf(o, "# n count_h count_v count_min count_max count_cross\n");
            for (size_t k = 0; k < H; k++)
                if (h[k] | h[H + k] | h[2 * H + k] | h[3 * H + k] | h[4 * H + k])
                    fprintf(o, "%zu %llu %llu %llu %llu %llu\n", k, h[k], h[H + k], h[2 * H + k], h[3 * H + k], h[4 * H + k]);
            fclose(o);
        }
    }
    printf("PERF %s mode=%s Lx=%d Ly=%d N=%d threads=%d pair_samples=%lld wall=%.2f per_s=%.4g\n", prefix.c_str(),
           A.mode ? "bond" : "site", A.Lx, A.Ly, A.Nobj, nthreads, M, wall, M / wall);
    return 0;
}
