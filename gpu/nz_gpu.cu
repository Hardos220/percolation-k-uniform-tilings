// Newman-Ziff site/bond percolation on many independent Lx x Ly tori at once:
// one torus per GPU thread.  Same algorithm and output format as nz/nz.c:
// for each sample, occupation numbers of the first winding along T1 (h), along
// T2 (v) and of the first cross cluster (x); the sample stops at the cross.
//
// Node = 8 bytes: par (root: -size, else parent index) and a 16-bit vector that
// is the displacement to the parent for a child and the first winding vector of
// the cluster for a root.  Requires Lx*Ly*nv small enough for int16 displacements.
//
// usage: nz_gpu CELLFILE site|bond Lx Ly SECONDS SEED OUTFILE [THREADS]
// Builds with nvcc (GPU) or with a C++ compiler (serial CPU check: -DCPU_CHECK).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#define DEV __device__
#else
#define HD
#define DEV
#endif

struct Node {
    int32_t par;
    int16_t dx, dy;
};

struct Lat {
    int nv, ne, Lx, Ly, mode;  // mode 0 site, 1 bond
    int32_t N, Nobj;
    const int *eu, *ew, *ea, *eb;              // cell edges
    const int *aoff, *aw, *aa, *ab;            // per-orbit adjacency (CSR)
};

static const int32_t EMPTY = INT32_MIN;

struct Rng {  // xoshiro128** (32-bit state words, cheap on GPU)
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

// root of i with path halving; (sx, sy) = displacement from i to the root; R = root node.
// Every parent index is bounds-checked (returns -1 on corruption): on a shared GPU an
// illegal access would bring down the MPS clients of other users too.
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

// join s and t, t = s + (ex, ey) unwrapped; flags: 1 winding T1, 2 winding T2, 4 cross, 8 corrupt
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
    if (Rs.par <= Rt.par) {  // r1 larger
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

// one sample on one torus; returns stop step; nh, nvv, nx = first-event occupation numbers
HD inline int32_t one_sample(const Lat &L, Node *nd, int32_t *order, Rng &rng,
                             int32_t &nh, int32_t &nvv, int32_t &nx) {
    Node init; init.dx = 0; init.dy = 0; init.par = L.mode ? -1 : EMPTY;
    for (int32_t i = 0; i < L.N; i++) nd[i] = init;
    nh = -1; nvv = -1; nx = -1;
    int32_t n;
    for (n = 0; n < L.Nobj; n++) {
        uint32_t j = (uint32_t)n + rng.bounded((uint32_t)(L.Nobj - n));
        int32_t x = order[j]; order[j] = order[n]; order[n] = x;
        if ((uint32_t)x >= (uint32_t)L.Nobj) { nx = -2; break; }
        int w = 0;
        if (L.mode) {
            int32_t c = x / L.ne, k = x - c * L.ne;
            int32_t a = c % L.Lx, b = c / L.Lx;
            int da = L.ea[k], db = L.eb[k];
            int a2 = a + da, b2 = b + db;
            a2 = (a2 % L.Lx + L.Lx) % L.Lx; b2 = (b2 % L.Ly + L.Ly) % L.Ly;
            w = join(nd, L.N, c * L.nv + L.eu[k], (b2 * L.Lx + a2) * L.nv + L.ew[k], da, db);
        } else {
            Node r; r.par = -1; r.dx = 0; r.dy = 0;
            nd[x] = r;
            int32_t c = x / L.nv, u = x - c * L.nv;
            int32_t a = c % L.Lx, b = c / L.Lx;
            for (int q = L.aoff[u]; q < L.aoff[u + 1]; q++) {
                int da = L.aa[q], db = L.ab[q];
                int a2 = a + da, b2 = b + db;
                a2 = (a2 % L.Lx + L.Lx) % L.Lx; b2 = (b2 % L.Ly + L.Ly) % L.Ly;
                int32_t y = (b2 * L.Lx + a2) * L.nv + L.aw[q];
                if ((uint32_t)y >= (uint32_t)L.N) { w |= 8; break; }
                if (nd[y].par != EMPTY) w |= join(nd, L.N, x, y, da, db);
            }
        }
        if (w & 8) { nx = -2; break; }
        if (w) {
            if ((w & 1) && nh < 0) nh = n + 1;
            if ((w & 2) && nvv < 0) nvv = n + 1;
            if (w & 4) { nx = n + 1; n++; break; }
        }
    }
    return n;
}

#ifdef __CUDACC__
typedef unsigned long long ull;

__global__ void kernel(Lat L, Node *nodes, int32_t *orders, ull *hist, uint64_t seed, int launch,
                       int samples, int nthreads, int *bad) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nthreads) return;
    Node *nd = nodes + (size_t)t * L.N;
    int32_t *order = orders + (size_t)t * L.Nobj;
    Rng rng;
    rng.seed(seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1)) ^ (0xD1B54A32D192ED03ULL * (uint64_t)(launch + 1)));
    size_t H = (size_t)L.Nobj + 1;
    for (int s = 0; s < samples; s++) {
        int32_t nh, nvv, nx;
        one_sample(L, nd, order, rng, nh, nvv, nx);
        if (nx < 0 || nh < 0 || nvv < 0) { atomicAdd(bad, 1); continue; }
        atomicAdd(&hist[0 * H + nh], 1ULL);
        atomicAdd(&hist[1 * H + nvv], 1ULL);
        atomicAdd(&hist[2 * H + (nh < nvv ? nh : nvv)], 1ULL);
        atomicAdd(&hist[3 * H + (nh > nvv ? nh : nvv)], 1ULL);
        atomicAdd(&hist[4 * H + nx], 1ULL);
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

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: nz_gpu CELLFILE site|bond Lx Ly SECONDS SEED OUTFILE [THREADS]\n");
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    int nv, ne;
    if (fscanf(f, "%d %d", &nv, &ne) != 2) return 1;
    std::vector<int> eu(ne), ew(ne), ea(ne), eb(ne);
    int maxoff = 1;
    for (int k = 0; k < ne; k++) {
        if (fscanf(f, "%d %d %d %d", &eu[k], &ew[k], &ea[k], &eb[k]) != 4) return 1;
        maxoff = std::max(maxoff, std::max(abs(ea[k]), abs(eb[k])));
    }
    fclose(f);
    std::vector<int> aoff(nv + 1, 0), aw(2 * ne), aa(2 * ne), ab(2 * ne);
    for (int k = 0; k < ne; k++) { aoff[eu[k] + 1]++; aoff[ew[k] + 1]++; }
    for (int u = 0; u < nv; u++) aoff[u + 1] += aoff[u];
    {
        std::vector<int> fill(aoff.begin(), aoff.end() - 1);
        for (int k = 0; k < ne; k++) {
            int q = fill[eu[k]]++; aw[q] = ew[k]; aa[q] = ea[k]; ab[q] = eb[k];
            q = fill[ew[k]]++; aw[q] = eu[k]; aa[q] = -ea[k]; ab[q] = -eb[k];
        }
    }
    Lat L;
    L.nv = nv; L.ne = ne; L.mode = strcmp(argv[2], "bond") == 0;
    L.Lx = atoi(argv[3]); L.Ly = atoi(argv[4]);
    L.N = L.Lx * L.Ly * nv;
    L.Nobj = L.mode ? L.Lx * L.Ly * ne : L.N;
    double seconds = atof(argv[5]);
    uint64_t seed = strtoull(argv[6], NULL, 10);
    if ((int64_t)L.Lx * L.Ly * maxoff >= 32000 || L.Lx >= 16000 || L.Ly >= 16000) {
        fprintf(stderr, "torus too large for int16 displacements\n");
        return 1;
    }
    size_t H = (size_t)L.Nobj + 1;
    std::vector<unsigned long long> hist(5 * H, 0);
    long long M = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    int nthreads = 0;
#ifdef __CUDACC__
    size_t per = (size_t)L.N * sizeof(Node) + (size_t)L.Nobj * sizeof(int32_t);
    const char *gm = getenv("GPUQ_GPU_MEM_GB");
    double budget = (gm ? atof(gm) : 4.0) - 1.5;
    if (budget < 0.5) budget = 0.5;
    nthreads = argc > 8 ? atoi(argv[8]) : 131072;
    size_t cap = (size_t)(budget * 1073741824.0 * 0.85) / per;
    if ((size_t)nthreads > cap) nthreads = (int)cap;
    nthreads = nthreads / 128 * 128;
    if (nthreads < 128) { fprintf(stderr, "not enough GPU memory\n"); return 1; }
    L.eu = upload(eu); L.ew = upload(ew); L.ea = upload(ea); L.eb = upload(eb);
    L.aoff = upload(aoff); L.aw = upload(aw); L.aa = upload(aa); L.ab = upload(ab);
    Node *nodes; int32_t *orders; ull *dh; int *dbad;
    CK(cudaMalloc(&nodes, per * nthreads));
    orders = (int32_t *)((char *)nodes + (size_t)L.N * sizeof(Node) * nthreads);
    CK(cudaMalloc(&dh, 5 * H * sizeof(ull)));
    CK(cudaMemset(dh, 0, 5 * H * sizeof(ull)));
    CK(cudaMalloc(&dbad, sizeof(int)));
    CK(cudaMemset(dbad, 0, sizeof(int)));
    init_orders<<<1024, 256>>>(orders, L.Nobj, nthreads);
    CK(cudaDeviceSynchronize());
    int samples = 1, launch = 0;
    double tl = 0;
    while (true) {
        double s0 = elapsed();
        kernel<<<nthreads / 128, 128>>>(L, nodes, orders, dh, seed, launch++, samples, nthreads, dbad);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        tl = elapsed() - s0;
        M += (long long)samples * nthreads;
        double left = seconds - elapsed();
        if (left <= 0) break;
        // aim at ~2 s per launch, never past the budget
        double per_sample = tl / samples;
        int want = (int)(std::min(2.0, left) / per_sample);
        samples = std::max(1, std::min(want, samples * 4));
        if (per_sample * samples > left * 1.05 && samples > 1) samples = std::max(1, (int)(left / per_sample));
    }
    CK(cudaMemcpy(hist.data(), dh, 5 * H * sizeof(ull), cudaMemcpyDeviceToHost));
    int bad = 0;
    CK(cudaMemcpy(&bad, dbad, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad) { fprintf(stderr, "%d samples without cross\n", bad); return 2; }
#else
    // serial CPU check of the same code path: a few "threads" one after another
    L.eu = eu.data(); L.ew = ew.data(); L.ea = ea.data(); L.eb = eb.data();
    L.aoff = aoff.data(); L.aw = aw.data(); L.aa = aa.data(); L.ab = ab.data();
    nthreads = argc > 8 ? atoi(argv[8]) : 4;
    const char *rp = getenv("REPLAY");  // e.g. "1,4": replay GPU threads [0, nthreads) launch by launch
    if (rp) {
        std::vector<int> sched;
        for (const char *q = rp; *q;) { sched.push_back(atoi(q)); while (*q && *q != ',') q++; if (*q) q++; }
        std::vector<Node> nd(L.N);
        std::vector<int32_t> order(L.Nobj);
        long long badc = 0;
        for (int t = 0; t < nthreads; t++) {
            for (int32_t i = 0; i < L.Nobj; i++) order[i] = i;
            for (size_t l = 0; l < sched.size(); l++) {
                Rng rng;
                rng.seed(seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1)) ^ (0xD1B54A32D192ED03ULL * (uint64_t)(l + 1)));
                for (int s = 0; s < sched[l]; s++) {
                    int32_t nh, nvv, nx;
                    one_sample(L, nd.data(), order.data(), rng, nh, nvv, nx);
                    if (nx < 0) badc++;
                    else { hist[4 * H + nx]++; M++; }
                }
            }
        }
        printf("REPLAY threads=%d schedule=%s samples=%lld bad=%lld\n", nthreads, rp, M, badc);
        return badc ? 2 : 0;
    }
    std::vector<Node> nodes((size_t)L.N * nthreads);
    std::vector<int32_t> orders((size_t)L.Nobj * nthreads);
    for (size_t i = 0; i < orders.size(); i++) orders[i] = (int32_t)(i % L.Nobj);
    int launch = 0;
    while (elapsed() < seconds) {
        for (int t = 0; t < nthreads; t++) {
            Rng rng;
            rng.seed(seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1)) ^ (0xD1B54A32D192ED03ULL * (uint64_t)(launch + 1)));
            for (int s = 0; s < 64; s++) {
                int32_t nh, nvv, nx;
                one_sample(L, &nodes[(size_t)t * L.N], &orders[(size_t)t * L.Nobj], rng, nh, nvv, nx);
                if (nx < 0) { fprintf(stderr, "no cross\n"); return 2; }
                hist[0 * H + nh]++; hist[1 * H + nvv]++;
                hist[2 * H + (nh < nvv ? nh : nvv)]++; hist[3 * H + (nh > nvv ? nh : nvv)]++;
                hist[4 * H + nx]++;
                M++;
            }
        }
        launch++;
    }
#endif
    double wall = elapsed();
    FILE *o = fopen(argv[7], "w");
    if (!o) { perror(argv[7]); return 1; }
    fprintf(o, "# cell %s mode %s Lx %d Ly %d nv %d ne %d N %lld samples %lld seed %s\n", argv[1],
            L.mode ? "bond" : "site", L.Lx, L.Ly, nv, ne, (long long)L.Nobj, M, argv[6]);
    fprintf(o, "# n count_h count_v count_min count_max count_cross\n");
    double steps = 0;
    for (size_t k = 0; k < H; k++) {
        steps += (double)k * hist[4 * H + k];
        if (hist[k] | hist[H + k] | hist[2 * H + k] | hist[3 * H + k] | hist[4 * H + k])
            fprintf(o, "%zu %llu %llu %llu %llu %llu\n", k, hist[k], hist[H + k], hist[2 * H + k],
                    hist[3 * H + k], hist[4 * H + k]);
    }
    fclose(o);
    printf("PERF cell=%s mode=%s Lx=%d Ly=%d N=%d threads=%d samples=%lld wall=%.2f samples_per_s=%.4g steps_per_s=%.4g\n",
           argv[1], L.mode ? "bond" : "site", L.Lx, L.Ly, L.Nobj, nthreads, M, wall, M / wall, steps / wall);
    return 0;
}
