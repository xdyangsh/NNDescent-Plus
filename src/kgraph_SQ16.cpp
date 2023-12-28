#ifndef KGRAPH_VERSION
#define KGRAPH_VERSION unknown
#endif
#ifndef KGRAPH_BUILD_NUMBER
#define KGRAPH_BUILD_NUMBER 
#endif
#ifndef KGRAPH_BUILD_ID
#define KGRAPH_BUILD_ID
#endif
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#define STRINGIFY_HELPER(x) #x
static char const *kgraph_version = STRINGIFY(KGRAPH_VERSION) "-" STRINGIFY(KGRAPH_BUILD_NUMBER) "," STRINGIFY(KGRAPH_BUILD_ID);

#ifdef _OPENMP
#include <omp.h>
#endif
#include <unordered_set>
#include <mutex>
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <boost/timer/timer.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include <boost/dynamic_bitset.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include "boost/smart_ptr/detail/spinlock.hpp"
#include "kgraph.h"
#ifndef MemoryUtils
#include "MemoryUtils.h"
#endif

namespace kgraph {

    using namespace std;
    using namespace boost;
    using namespace boost::accumulators;

    unsigned verbosity = default_verbosity;

    typedef boost::detail::spinlock Lock;
    typedef std::lock_guard<Lock> LockGuard;

    // generate size distinct random numbers < N
    template <typename RNG>
    static void GenRandom (RNG &rng, unsigned *addr, unsigned size, unsigned N) {
        if (N == size) {
            for (unsigned i = 0; i < size; ++i) {
                addr[i] = i;
            }
            return;
        }
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i-1]) {
                addr[i] = addr[i-1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    struct Neighbor {
        uint32_t id;
        dist_type dist;
        bool flag;  // whether this entry is a newly found one
        Neighbor () {}
        Neighbor (unsigned i, dist_type d, bool f = true): id(i), dist(d), flag(f) {
        }
    };

    // extended neighbor structure for search time
    struct NeighborX: public Neighbor {
        uint16_t m;
        uint16_t M; // actual M used
        NeighborX () {}
        NeighborX (unsigned i, dist_type d): Neighbor(i, d, true), m(0), M(0) {
        }
    };

    static inline bool operator < (Neighbor const &n1, Neighbor const &n2) {
        return n1.dist < n2.dist;
    }

    static inline bool operator == (Neighbor const &n1, Neighbor const &n2) {
        return n1.id == n2.id;
    }

    typedef vector<Neighbor> Neighbors;

    // both pool and knn should be sorted in ascending order
    static float EvaluateRecall (Neighbors const &pool, Neighbors const &knn) {
        if (knn.empty()) return 1.0;
        unsigned found = 0;
        unsigned n_p = 0;
        unsigned n_k = 0;
        for (;;) {
            if (n_p >= pool.size()) break;
            if (n_k >= knn.size()) break;
            if (knn[n_k].dist < pool[n_p].dist) {
                ++n_k;
            }
            else if (knn[n_k].dist == pool[n_p].dist) {
                ++found;
                ++n_k;
                ++n_p;
            }
            else {
                cerr << "Distance is unstable." << endl;
                cerr << "Exact";
                for (auto const &p: knn) {
                    cerr << ' ' << p.id << ':' << p.dist;
                }
                cerr << endl;
                cerr << "Approx";
                for (auto const &p: pool) {
                    cerr << ' ' << p.id << ':' << p.dist;
                }
                cerr << endl;
                throw runtime_error("distance is unstable");
            }
        }
        return float(found) / knn.size();
    }
/*
    static float EvaluateAccuracy (Neighbors const &pool, Neighbors const &knn) {
        unsigned m = std::min(pool.size(), knn.size());
        float sum = 0;
        unsigned cnt = 0;
        for (unsigned i = 0; i < m; ++i) {
            if (knn[i].dist >  0) {
                sum += abs(pool[i].dist - knn[i].dist) / knn[i].dist;
                ++cnt;
            }
        }
        return cnt > 0 ? sum / cnt: 0;
    }
*/
    static float EvaluateOneRecall (Neighbors const &pool, Neighbors const &knn) {
        if (pool[0].dist == knn[0].dist) return 1.0;
        return 0;
    }

    static float EvaluateDelta (Neighbors const &pool, unsigned K) {
        unsigned c = 0;
        unsigned N = K;
        if (pool.size() < N) N = pool.size();
        for (unsigned i = 0; i < N; ++i) {
            if (pool[i].flag) ++c;
        }
        return float(c) / K;
    }

    struct Control {
        unsigned id;
        Neighbors neighbors;
    };

    // try insert nn into the list
    // the array addr must contain at least K+1 entries:
    //      addr[0..K-1] is a sorted list
    //      addr[K] is as output parameter
    // * if nn is already in addr[0..K-1], return K+1
    // * Otherwise, do the equivalent of the following
    //      put nn into addr[K]
    //      make addr[0..K] sorted
    //      return the offset of nn's index in addr (could be K)
    //
    // Special case:  K == 0
    //      addr[0] <- nn
    //      return 0
    template <typename NeighborT>
    unsigned UpdateKnnListHelper (NeighborT *addr, unsigned K, NeighborT nn) {
        // find the location to insert
        unsigned j;
        unsigned i = K;
        while (i > 0) {
            j = i - 1;
            if (addr[j].dist <= nn.dist) break;
            i = j;
        }
        // check for equal ID
        unsigned l = i;
        while (l > 0) {
            j = l - 1;
            if (addr[j].dist < nn.dist) break;
            if (addr[j].id == nn.id) return K + 1;
            l = j;
        }
        // i <= K-1
        j = K;
        while (j > i) {
            addr[j] = addr[j-1];
            --j;
        }
        addr[i] = nn;
        return i;
    }

    static inline unsigned UpdateKnnList (Neighbor *addr, unsigned K, Neighbor nn) {
        return UpdateKnnListHelper<Neighbor>(addr, K, nn);
    }

    static inline unsigned UpdateKnnList (NeighborX *addr, unsigned K, NeighborX nn) {
        return UpdateKnnListHelper<NeighborX>(addr, K, nn);
    }

    void LinearSearch (IndexOracle const &oracle, unsigned i, unsigned K, vector<Neighbor> *pnns) {
        vector<Neighbor> nns(K+1);
        unsigned N = oracle.size();
        Neighbor nn;
        nn.id = 0;
        nn.flag = true; // we don't really use this
        unsigned k = 0;
        while (nn.id < N) {
            if (nn.id != i) {
                nn.dist = oracle(i, nn.id);
                UpdateKnnList(&nns[0], k, nn);
                if (k < K) ++k;
            }
            ++nn.id;
        }
        nns.resize(K);
        pnns->swap(nns);
    }

    unsigned SearchOracle::search (unsigned K, float epsilon, unsigned *ids, dist_type *dists) const {
        vector<Neighbor> nns(K+1);
        unsigned N = size();
        unsigned L = 0;
        for (unsigned k = 0; k < N; ++k) {
            dist_type k_dist = operator () (k);
            if (k_dist > epsilon) continue;
            UpdateKnnList(&nns[0], L, Neighbor(k, k_dist));
            if (L < K) ++L;
        }
        if (ids) {
            for (unsigned k = 0; k < L; ++k) {
                ids[k] = nns[k].id;
            }
        }
        if (dists) {
            for (unsigned k = 0; k < L; ++k) {
                dists[k] = nns[k].dist;
            }
        }
        return L;
    }

    void GenerateControl (IndexOracle const &oracle, unsigned C, unsigned K, vector<Control> *pcontrols) {
        vector<Control> controls(C);
        {
            vector<unsigned> index(oracle.size());
            int i = 0;
            for (unsigned &v: index) {
                v = i++;
            }
            random_shuffle(index.begin(), index.end());
#pragma omp parallel for 
            for (unsigned i = 0; i < C; ++i) {
                controls[i].id = index[i];
                LinearSearch(oracle, index[i], K, &controls[i].neighbors);
            }
        }
        pcontrols->swap(controls);
    }

    static char const *KGRAPH_MAGIC = "KNNGRAPH";
    static unsigned constexpr KGRAPH_MAGIC_SIZE = 8;
    static uint32_t constexpr SIGNATURE_VERSION = 2;

    class KGraphImpl: public KGraph {
    protected:
        vector<unsigned> M;
        vector<vector<Neighbor>> graph;
        bool no_dist;   // Distance & flag information in Neighbor is not valid.


        // actual M for a node that should be used in search time
        unsigned actual_M (unsigned pM, unsigned i) const {
            return std::min(std::max(M[i], pM), unsigned(graph[i].size()));
        }

    public:
        virtual ~KGraphImpl () {
        }
        virtual void load (char const *path) {
            static_assert(sizeof(unsigned) == sizeof(uint32_t), "unsigned must be 32-bit");
            ifstream is(path, ios::binary);
            char magic[KGRAPH_MAGIC_SIZE];
            uint32_t sig_version;
            uint32_t sig_cap;
            uint32_t N;
            is.read(magic, sizeof(magic));
            is.read(reinterpret_cast<char *>(&sig_version), sizeof(sig_version));
            is.read(reinterpret_cast<char *>(&sig_cap), sizeof(sig_cap));
            if (sig_version != SIGNATURE_VERSION) throw runtime_error("data version not supported.");
            is.read(reinterpret_cast<char *>(&N), sizeof(N));
            if (!is) runtime_error("error reading index file.");
            for (unsigned i = 0; i < KGRAPH_MAGIC_SIZE; ++i) {
                if (KGRAPH_MAGIC[i] != magic[i]) runtime_error("index corrupted.");
            }
            no_dist = sig_cap & FORMAT_NO_DIST;
            graph.resize(N);
            M.resize(N);
            vector<uint32_t> nids;
            for (unsigned i = 0; i < graph.size(); ++i) {
                auto &knn = graph[i];
                unsigned K;
                is.read(reinterpret_cast<char *>(&M[i]), sizeof(M[i]));
                is.read(reinterpret_cast<char *>(&K), sizeof(K));
                if (!is) runtime_error("error reading index file.");
                knn.resize(K);
                if (no_dist) {
                    nids.resize(K);
                    is.read(reinterpret_cast<char *>(&nids[0]), K * sizeof(nids[0]));
                    for (unsigned k = 0; k < K; ++k) {
                        knn[k].id = nids[k];
                        knn[k].dist = 0;
                        knn[k].flag = false;
                    }
                }
                else {
                    is.read(reinterpret_cast<char *>(&knn[0]), K * sizeof(knn[0]));
                }
            }
        }

        virtual void save (char const *path, int format) const {
            ofstream os(path, ios::binary);
            uint32_t N = graph.size();
            vector<unsigned> nids;
            for (unsigned i = 0; i < graph.size(); ++i) {
                auto const &knn = graph[i];
                uint32_t K = knn.size();
                nids.resize(K);
                for (unsigned k = 0; k < K; ++k) {
                    nids[k] = knn[k].id;
                    if(i==0)
                    {
                        std::cout<<nids[k]<<" ";
                    }
                }
                os.write(reinterpret_cast<char const *>(&nids[0]), K * sizeof(nids[0]));
            }
            std::cout<<std::endl;
            os.close();
        }

        virtual void build (IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,char const *path,char const *truth_path);
        virtual void save_groundtruth(IndexOracle const &oracle,char const *path =nullptr);

        /*
        virtual void prune (unsigned K) {
            for (auto &v: graph) {
                if (v.size() > K) {
                    v.resize(K);
                }
            }
            return;
            vector<vector<unsigned>> pruned(graph.size());
            vector<set<unsigned>> reachable(graph.size());
            vector<bool> added(graph.size());
            for (unsigned k = 0; k < K; ++k) {
#pragma omp parallel for
                for (unsigned n = 0; n < graph.size(); ++n) {
                    vector<unsigned> const &from = graph[n];
                    if (from.size() <= k) continue;
                    unsigned e = from[k];
                    if (reachable[n].count(e)) {
                        added[n] = false;
                    }
                    else {
                        pruned[n].push_back(e);
                        added[n] = true;
                    }
                }
                // expand reachable
#pragma omp parallel for
                for (unsigned n = 0; n < graph.size(); ++n) {
                    vector<unsigned> const &to = pruned[n];
                    set<unsigned> &nn = reachable[n];
                    if (added[n]) {
                        for (unsigned v: pruned[to.back()]) {
                            nn.insert(v);
                        }
                    }
                    for (unsigned v: to) {
                        if (added[v]) {
                            nn.insert(pruned[v].back());
                        }
                    }
                }
            }
            graph.swap(pruned);
        }
        */

        virtual unsigned search (SearchOracle const &oracle, SearchParams const &params, unsigned *ids, dist_type *dists, SearchInfo *pinfo) const {
            if (graph.size() > oracle.size()) {
                throw runtime_error("dataset larger than index");
            }
            if (params.P >= graph.size()) {
                if (pinfo) {
                    pinfo->updates = 0;
                    pinfo->cost = 1.0;
                }
                return oracle.search(params.K, params.epsilon, ids, dists);
            }
            vector<NeighborX> knn(params.K + params.P +1);
            vector<NeighborX> results;
            // flags access is totally random, so use small block to avoid
            // extra memory access
            boost::dynamic_bitset<> flags(graph.size(), false);

            if (params.init && params.T > 1) {
                throw runtime_error("when init > 0, T must be 1.");
            }

            unsigned seed = params.seed;
            unsigned updates = 0;
            if (seed == 0) seed = time(NULL);
            mt19937 rng(seed);
            unsigned n_comps = 0;
            for (unsigned trial = 0; trial < params.T; ++trial) {
                unsigned L = params.init;
                if (L == 0) {   // generate random starting points
                    vector<unsigned> random(params.P);
                    GenRandom(rng, &random[0], random.size(), graph.size());
                    for (unsigned s: random) {
                        if (!flags[s]) {
                            knn[L++].id = s;
                            //flags[s] = true;
                        }
                    }
                }
                else {          // user-provided starting points.
                    if (!ids) throw invalid_argument("no initial data provided via ids");
                    if (!(L < params.K)) throw invalid_argument("L < params.K");
                    for (unsigned l = 0; l < L; ++l) {
                        knn[l].id = ids[l];
                    }
                }
                for (unsigned k = 0; k < L; ++k) {
                    auto &e = knn[k];
                    flags[e.id] = true;
                    e.flag = true;
                    e.dist = oracle(e.id);
                    e.m = 0;
                    e.M = actual_M(params.M, e.id);
                }
                sort(knn.begin(), knn.begin() + L);

                unsigned k =  0;
                while (k < L) {
                    auto &e = knn[k];
                    if (!e.flag) { // all neighbors of this node checked
                        ++k;
                        continue;
                    }
                    unsigned beginM = e.m;
                    unsigned endM = beginM + params.S; // check this many entries
                    if (endM > e.M) {   // we are done with this node
                        e.flag = false;
                        endM = e.M;
                    }
                    e.m = endM;
                    // all modification to knn[k] must have been done now,
                    // as we might be relocating knn[k] in the loop below
                    auto const &neighbors = graph[e.id];
                    for (unsigned m = beginM; m < endM; ++m) {
                        unsigned id = neighbors[m].id;
                        //BOOST_VERIFY(id < graph.size());
                        if (flags[id]) continue;
                        flags[id] = true;
                        ++n_comps;
                        dist_type dist = oracle(id);
                        NeighborX nn(id, dist);
                        unsigned r = UpdateKnnList(&knn[0], L, nn);
                        BOOST_VERIFY(r <= L);
                        //if (r > L) continue;
                        if (L + 1 < knn.size()) ++L;
                        if (r < L) {
                            knn[r].M = actual_M(params.M, id);
                            if (r < k) {
                                k = r;
                            }
                        }
                    }
                }
                if (L > params.K) L = params.K;
                if (results.empty()) {
                    results.reserve(params.K + 1);
                    results.resize(L + 1);
                    copy(knn.begin(), knn.begin() + L, results.begin());
                }
                else {
                    // update results
                    for (unsigned l = 0; l < L; ++l) {
                        unsigned r = UpdateKnnList(&results[0], results.size() - 1, knn[l]);
                        if (r < results.size() /* inserted */ && results.size() < (params.K + 1)) {
                            results.resize(results.size() + 1);
                        }
                    }
                }
            }
            results.pop_back();
            // check epsilon
            {
                for (unsigned l = 0; l < results.size(); ++l) {
                    if (results[l].dist > params.epsilon) {
                        results.resize(l);
                        break;
                    }
                }
            }
            unsigned L = results.size();
            /*
            if (!(L <= params.K)) {
                cerr << L << ' ' << params.K << endl;
            }
            */
            if (!(L <= params.K)) throw runtime_error("L <= params.K");
            // check epsilon
            if (ids) {
                for (unsigned k = 0; k < L; ++k) {
                    ids[k] = results[k].id;
                }
            }
            if (dists) {
                for (unsigned k = 0; k < L; ++k) {
                    dists[k] = results[k].dist;
                }
            }
            if (pinfo) {
                pinfo->updates = updates;
                pinfo->cost = float(n_comps) / graph.size();
            }
            return L;
        }

        virtual void get_nn (unsigned id, unsigned *nns, dist_type *dist, unsigned *pM, unsigned *pL) const {
            if (!(id < graph.size())) throw invalid_argument("id too big");
            auto const &v = graph[id];
            *pM = M[id];
            *pL = v.size();
            if (nns) {
                for (unsigned i = 0; i < v.size(); ++i) {
                    nns[i] = v[i].id;
                }
            }
            if (dist) {
                if (no_dist) throw runtime_error("distance information is not available");
                for (unsigned i = 0; i < v.size(); ++i) {
                    dist[i] = v[i].dist;
                }
            }
        }

        void prune1 () {
            for (unsigned i = 0; i < graph.size(); ++i) {
                if (graph[i].size() > M[i]) {
                    graph[i].resize(M[i]);
                }
            }
        }


        void prune2 () {
            vector<vector<unsigned>> reverse(graph.size()); // reverse of new graph
            vector<unsigned> new_L(graph.size(), 0);
            unsigned L = 0;
            unsigned total = 0;
            for (unsigned i = 0; i < graph.size(); ++i) {
                if (M[i] > L) L = M[i];
                total += M[i];
                for (auto &e: graph[i]) {
                    e.flag = false;             // haven't been visited yet
                }
            }
            progress_display progress(total, cerr);
            vector<unsigned> todo(graph.size());
            for (unsigned i = 0; i < todo.size(); ++i) todo[i] = i;
            vector<unsigned> new_todo(graph.size());
            for (unsigned l = 0; todo.size(); ++l) {
                BOOST_VERIFY(l <= L);
                new_todo.clear();
                for (unsigned i: todo) {
                    if (l >= M[i]) continue;
                    new_todo.push_back(i);
                    auto &v = graph[i];
                    BOOST_VERIFY(l < v.size());
                    if (v[l].flag) continue; // we have visited this one already
                    v[l].flag = true;        // now we have seen this one
                    ++progress;
                    unsigned T;
                    {
                        auto &nl = new_L[i];
                        // shift the entry to add
                        T = v[nl].id = v[l].id;
                        v[nl].dist = v[l].dist;
                        ++nl;
                    }
                    reverse[T].push_back(i);
                    {
                        auto const &u = graph[T];
                        for (unsigned ll = l + 1; ll < M[i]; ++ll) {
                            if (v[ll].flag) continue;
                            for (unsigned j = 0; j < new_L[T]; ++j) { // new graph
                                if (v[ll].id == u[j].id) {
                                    v[ll].flag = true;
                                    ++progress;
                                    break;
                                }
                            }
                        }
                    }
                    {
                        for (auto r: reverse[i]) {
                            auto &u = graph[r];
                            for (unsigned ll = l; ll < M[r]; ++ll) {
                                // must start from l: as item l might not have been checked
                                // for reverse
                                if (u[ll].id == T) {
                                    if (!u[ll].flag) ++progress;
                                    u[ll].flag = true;
                                }
                            }
                        }
                    }
                }
                todo.swap(new_todo);
            }
            BOOST_VERIFY(progress.count() == total);
            M.swap(new_L);
            prune1();
        }

        virtual void prune (IndexOracle const &oracle, unsigned level) {
            if (level & PRUNE_LEVEL_1) {
                prune1();
            }
            if (level & PRUNE_LEVEL_2) {
                prune2();
            }
        }

        void reverse (int rev_k) {
            if (rev_k == 0) return;
            if (no_dist) throw runtime_error("Need distance information to reverse graph");
            {
                cerr << "Graph completion with reverse edges..." << endl;
                vector<vector<Neighbor>> ng(graph.size()); // new graph adds on original one
                //ng = graph;
                progress_display progress(graph.size(), cerr);
                for (unsigned i = 0; i < graph.size(); ++i) {
                    auto const &v = graph[i];
                    unsigned K = M[i];
                    if (rev_k > 0) {
                        K = rev_k;
                        if (K > v.size()) K = v.size();
                    }
                    //if (v.size() < XX) XX = v.size();
                    for (unsigned j = 0; j < K; ++j) {
                        auto const &e = v[j];
                        auto re = e;
                        re.id = i;
                        ng[i].push_back(e);
                        ng[e.id].push_back(re);
                    }
                    ++progress;
                }
                graph.swap(ng);
            }
            {
                cerr << "Reranking edges..." << endl;
                progress_display progress(graph.size(), cerr);
#pragma omp parallel for 
                for (unsigned i = 0; i < graph.size(); ++i) {
                    auto &v = graph[i];
                    std::sort(v.begin(), v.end());
                    v.resize(std::unique(v.begin(), v.end()) - v.begin());
                    M[i] = v.size();
#pragma omp critical
                    ++progress;
                }
            }
        }
    };

    class KGraphConstructor: public KGraphImpl {
        // The neighborhood structure maintains a pool of near neighbors of an object.
        // The neighbors are stored in the pool.  "n" (<="params.L") is the number of valid entries
        // in the pool, with the beginning "k" (<="n") entries sorted.
        struct Nhood { // neighborhood
            Lock lock;
            dist_type radius;   // distance of interesting range
            dist_type radiusM;
            Neighbors pool;
            unsigned L;     // # valid items in the pool,  L + 1 <= pool.size()
            unsigned M;     // we only join items in pool[0..M)
            bool found;     // helped found new NN in this round
            vector<unsigned> nn_old;
            vector<unsigned> nn_new;
            vector<unsigned> rnn_old;
            vector<unsigned> rnn_new;

            // only non-readonly method which is supposed to be called in parallel
            unsigned parallel_try_insert (unsigned id, dist_type dist) {
                if (dist > radius) return pool.size();
                LockGuard guard(lock);
                unsigned l = UpdateKnnList(&pool[0], L, Neighbor(id, dist, true));
                if (l <= L) { // inserted
                    if (L + 1 < pool.size()) { // if l == L + 1, there's a duplicate
                        ++L;
                    }
                    else {
                        radius = pool[L-1].dist;
                    }
                }
                return l;
            }

            // join should not be conflict with insert
            template <typename C>
            void join (C callback) const {
                for (unsigned const i: nn_new) {
                    for (unsigned const j: nn_new) {
                        if (i < j) {
                            callback(i, j);
                        }
                    }
                    for (unsigned j: nn_old) {
                        callback(i, j);
                    }
                }
            }
        };

        IndexOracle const &oracle;
        IndexParams params;
        IndexInfo *pinfo;
        vector<Nhood> nhoods;
        const char *truth_path;
        long FixedMemory;
        size_t n_comps;

        void init () {
            cout<<"init开始当前内存:"<<getCurrentMemoryUsage()<<endl;
            unsigned N = oracle.size();
            unsigned seed = params.seed;
            mt19937 rng(seed);
            for (auto &nhood: nhoods) {
                nhood.pool.resize(params.L+1);
                nhood.radius = numeric_limits<dist_type>::max();
            }
            cout << N * nhoods[0].pool.size() * sizeof(Neighbor) << std::endl;
            cout<<"初始化pool当前内存:"<<getCurrentMemoryUsage()<<endl;
            FixedMemory=getCurrentMemoryUsage();
            for (auto &nhood: nhoods) {
                nhood.nn_new.resize(params.S * 2);
            }
#pragma omp parallel 
            {
#ifdef _OPENMP
                mt19937 rng(seed ^ omp_get_thread_num());
#else
                mt19937 rng(seed);
#endif
                vector<unsigned> random(params.S + 1);
#pragma omp for
                for (unsigned n = 0; n < N; ++n) {
                    auto &nhood = nhoods[n];
                    Neighbors &pool = nhood.pool;
                    GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(), N);
                    GenRandom(rng, &random[0], random.size(), N);
                    nhood.L = params.S;
                    nhood.M = params.S;
                    unsigned i = 0;
                    for (unsigned l = 0; l < nhood.L; ++l) {
                        if (random[i] == n) ++i;
                        auto &nn = nhood.pool[l];
                        nn.id = random[i++];
                        nn.dist = oracle(nn.id, n);
                        nn.flag = true;
                    }
                    sort(pool.begin(), pool.begin() + nhood.L);
                }
            }
        }
        void join () {
            size_t cc = 0;
#pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+:cc) 
            for (unsigned n = 0; n < oracle.size(); ++n) {
                size_t uu = 0;
                nhoods[n].found = false;
                nhoods[n].join([&](unsigned i, unsigned j) {
                        dist_type dist = oracle(i, j);
                        ++cc;
                        unsigned r;
                        r = nhoods[i].parallel_try_insert(j, dist);
                        if (r < params.K) ++uu;
                        r=nhoods[j].parallel_try_insert(i, dist);
                        if (r < params.K) ++uu;
                });
                nhoods[n].found = uu > 0;
            }
            n_comps += cc;
        }
        void update () {
            unsigned N = oracle.size();
            for (auto &nhood: nhoods) {
                nhood.nn_new.clear();
                nhood.nn_old.clear();
                nhood.rnn_new.clear();
                nhood.rnn_old.clear();
                nhood.radius = nhood.pool.back().dist;
            }
            //!!! compute radius2
#pragma omp parallel for 
            for (unsigned n = 0; n < N; ++n) {
                auto &nhood = nhoods[n];
                if (nhood.found) {
                    unsigned maxl = std::min(nhood.M + params.S, nhood.L);
                    unsigned c = 0;
                    unsigned l = 0;
                    while ((l < maxl) && (c < params.S)) {
                        if (nhood.pool[l].flag) ++c;
                        ++l;
                    }
                    nhood.M = l;
                }
                BOOST_VERIFY(nhood.M > 0);
                nhood.radiusM = nhood.pool[nhood.M-1].dist;
            }
#pragma omp parallel for 
            for (unsigned n = 0; n < N; ++n) {
                auto &nhood = nhoods[n];
                auto &nn_new = nhood.nn_new;
                auto &nn_old = nhood.nn_old;
                for (unsigned l = 0; l < nhood.M; ++l) {
                    auto &nn = nhood.pool[l];
                    auto &nhood_o = nhoods[nn.id];  // nhood on the other side of the edge
                    if (nn.flag) {
                        nn_new.push_back(nn.id);
                        if (nn.dist > nhood_o.radiusM) {
                            LockGuard guard(nhood_o.lock);
                            nhood_o.rnn_new.push_back(n);
                        }
                        nn.flag = false;
                    }
                    else {
                        nn_old.push_back(nn.id);
                        if (nn.dist > nhood_o.radiusM) {
                            LockGuard guard(nhood_o.lock);
                            nhood_o.rnn_old.push_back(n);
                        }
                    }
                }
            }
            for (unsigned i = 0; i < N; ++i) {
                auto &nn_new = nhoods[i].nn_new;
                auto &nn_old = nhoods[i].nn_old;
                auto &rnn_new = nhoods[i].rnn_new;
                auto &rnn_old = nhoods[i].rnn_old;
                if (params.R && (rnn_new.size() > params.R)) {
                    random_shuffle(rnn_new.begin(), rnn_new.end());
                    rnn_new.resize(params.R);
                }
                nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
                if (params.R && (rnn_old.size() > params.R)) {
                    random_shuffle(rnn_old.begin(), rnn_old.end());
                    rnn_old.resize(params.R);
                }
                nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            }
        }

        void write_info(vector<float> &all_cost,vector<float> &all_recall,vector<float> &all_time,vector<long> &all_peakMemory,vector<float> &all_join,vector<float> &all_update,long &FixedMemory,string data)
        {
            string path;
            path=string("./result/short/")+string("kgraph_data(")+data+string(")_L")+std::to_string(params.L)
                +string("_S")+std::to_string(params.S)+string("_R")+std::to_string(params.R)+string(".csv");
            ofstream ofs(path,std::ios::out | std::ios::app);
            if (ofs.is_open()) 
            {
                for(int i=0;i<all_cost.size();++i)
                {
                    ofs<<"cost"<<",";
                    ofs<<"recall"<<",";
                    ofs<<"time"<<",";
                    ofs<<"peakmemory"<<",";
                    ofs<<"join"<<",";
                    ofs<<"update"<<",";
                }
                ofs<<"FixedMemory,"<<endl;
                for(int i=0;i<all_cost.size();++i)
                {
                    ofs<<all_cost[i]<<",";
                    ofs<<all_recall[i]<<",";
                    ofs<<all_time[i]<<",";
                    ofs<<all_peakMemory[i]<<",";
                    ofs<<all_join[i]<<",";
                    ofs<<all_update[i]<<",";
                }
                ofs<<FixedMemory<<endl<<endl;
                ofs.close();
            }
            cout<<endl;
        }
        void LoadSampleGroundtruth(std::vector<std::vector<uint32_t>> &knng,
                            std::vector<uint32_t> &ids,
                            const char *path) {
            std::ifstream ifs(path, std::ios::out | std::ios::binary);
            cout << "Loading ground truth from: " << path << endl;
            assert(ifs.is_open());

            uint32_t num_points;  // N
            ifs.read((char *)&num_points, sizeof(uint32_t));
            cout << "# of samples: " << num_points << endl;
            ifs.seekg(0, std::ios::end);
            std::ios::pos_type ss = ifs.tellg();
            size_t fsize = (size_t)ss;
            size_t num_dimensions=(fsize-sizeof(uint32_t))/num_points/sizeof(uint32_t)-1;
            cout<<"k of exact knn:  "<<num_dimensions<<endl;
            std::vector<uint32_t> buff(num_dimensions + 1);
            knng.resize(num_points);
            ids.resize(num_points);
            ifs.seekg(sizeof(uint32_t), std::ios::beg);
            int counter = 0;
            while (
                ifs.read((char *)buff.data(), (num_dimensions + 1) * sizeof(uint32_t))) {
                std::vector<uint32_t> row(num_dimensions);
                ids[counter] = static_cast<uint32_t>(buff[0]);
                for (int d = 1; d < num_dimensions + 1; d++) {
                row[d - 1] = static_cast<uint32_t>(buff[d]);
                }
                knng[counter++] = std::move(row);
            }
            ifs.close();
        }
        float computeOverallRecall(std::vector<std::vector<uint32_t>> &exact_knng,
                                  std::vector<uint32_t> &ids) {
            //assert(knn_vec.size() >= 10000);
            //assert(knn_vec[0].size() <= 100);
            
            
            int nk = params.K;
            size_t hit_case = 0;
#pragma omp parallel for reduction(+:hit_case)
            for (int i=0;i<10000;i++) {
                uint32_t sample_id=ids[i];
                //printf("%d\n", i);
                std::vector<int> ann;
                // get exact result
                std::vector<int> enn;
                enn.resize(nk);
                ann.resize(nk);

                for (int j=0;j<nk;j++)
                {
                    ann[j] = nhoods[sample_id].pool[j].id;
                    enn[j] = exact_knng[i][j];
                }
                    
                hit_case += computeSingleRecall(enn, ann);
            }
            float re = (float) hit_case / (float) 10000 / (float) nk;
            //std::cout << re<< std::endl;
            return re;
        }


        size_t computeSingleRecall(std::vector<int>& _enn, std::vector<int>& _ann) {
            size_t hit_case = 0;
            int nk = _ann.size();

            for (int i=0;i<nk;i++) {
                int annid = _ann[i];
                for (int j=0;j<nk;j++) {
                    if (_enn[j] == annid) {
                        hit_case++;
                        break;
                    }
                }
            }

            return hit_case;
        }


public:
        void save_knng (char const *path) const {
            ofstream os(path, ios::binary);
            uint32_t N = oracle.size();
            vector<unsigned> nids;
            for (unsigned i = 0; i < N; ++i) {
                auto const &pool = nhoods[i].pool;
                unsigned K = params.K;
                nids.resize(K);
                for (unsigned k = 0; k < K; ++k) {
                    nids[k] = pool[k].id;
                    /*
                    if(i==0)
                    {
                        std::cout<<nids[k]<<" ";
                    }*/
                }
                os.write(reinterpret_cast<char const *>(&nids[0]), K * sizeof(nids[0]));
            }
           // std::cout<<std::endl;
            os.close();
        }

        KGraphConstructor (IndexOracle const &o, IndexParams const &p, IndexInfo *r, const char *truth_path)
            : oracle(o), params(p), pinfo(r), nhoods(o.size()), n_comps(0),truth_path(truth_path)
        {
            no_dist = false;
            cout<<"读取groundtruth前当前内存:"<<getCurrentMemoryUsage()<<endl;
            vector<vector<uint32_t>> ground_truth_knng;
            vector<uint32_t> ground_truth_ids;
            LoadSampleGroundtruth(ground_truth_knng, ground_truth_ids, truth_path);
            cout<<"读取groundtruth后当前内存:"<<getCurrentMemoryUsage()<<endl;
            unsigned N = oracle.size();
            if (N <= params.K) throw runtime_error("K larger than dataset size");
            if (N < params.controls) {
                cerr << "Warning: small dataset, shrinking control size to " << N << "." << endl;
                params.controls = N;
            }
            if (N <= params.L) {
                cerr << "Warning: small dataset, shrinking L to " << (N-1) << "." << endl;
                params.L = N - 1; 
            }
            if (N <= params.S) {
                cerr << "Warning: small dataset, shrinking S to " << (N-1) << "." << endl;
                params.S = N - 1; 
            }
            cout<<"当前内存:"<<getCurrentMemoryUsage()<<endl;
            vector<Control> controls;
            if (verbosity > 0) cerr << "Generating control..." << endl;
            GenerateControl(oracle, params.controls, params.K, &controls);
            cout<<"当前内存:"<<getCurrentMemoryUsage()<<endl;
            if (verbosity > 0) cerr << "Initializing..." << endl;
            // initialize nhoods
            boost::timer::cpu_timer timer;
            init();

            // iterate until converge
            float total = N * float(N - 1) / 2;
            IndexInfo info;
            info.stop_condition = IndexInfo::ITERATION;
            info.recall = 0;
            info.accuracy = numeric_limits<float>::max();
            info.cost = 0;
            info.iterations = 0;
            info.delta = 1.0;
            info.time = 0;
            vector<float> all_cost(params.iterations,0);
            vector<float> all_recall(params.iterations,0);
            vector<float> all_time(params.iterations,0);
            vector<long> all_peakMemory(params.iterations,0);
            vector<float> all_join(params.iterations,0);
            vector<float> all_update(params.iterations,0);
            for (unsigned it = 0; (params.iterations <= 0) || (it < params.iterations); ++it) {
                ++info.iterations;
                auto times1 = timer.elapsed();
                join();
                auto times2 = timer.elapsed();
                float join_time=times2.wall / 1e9-times1.wall / 1e9;
                all_join[it] = join_time;
                /*
                std::cout<<"iter:"<<it+1<<"  start time(s): "<<times1.wall / 1e9<<std::endl;
                std::cout<<"iter:"<<it+1<<"  join time(s): "<<times2.wall / 1e9-times1.wall / 1e9<<std::endl;
                */
                {
                    info.cost = n_comps / total;
                    accumulator_set<float, stats<tag::mean>> one_exact;
                    accumulator_set<float, stats<tag::mean>> one_approx;
                    accumulator_set<float, stats<tag::mean>> one_recall;
                    accumulator_set<float, stats<tag::mean>> recall;
                    accumulator_set<float, stats<tag::mean>> accuracy;
                    accumulator_set<float, stats<tag::mean>> M;
                    accumulator_set<float, stats<tag::mean>> delta;
                    for (auto const &nhood: nhoods) {
                        M(nhood.M);
                        delta(EvaluateDelta(nhood.pool, params.K));
                    }
                    /**/
                    for (auto const &c: controls) {
                        one_approx(nhoods[c.id].pool[0].dist);
                        one_exact(c.neighbors[0].dist);
                        one_recall(EvaluateOneRecall(nhoods[c.id].pool, c.neighbors));
                        recall(EvaluateRecall(nhoods[c.id].pool, c.neighbors));
                        //accuracy(EvaluateAccuracy(nhoods[c.id].pool, c.neighbors));
                    }
                    // float sum_recall = 0.0;
                    // for (unsigned i = 0; i < ground_truth_ids.size(); i++) {
                    //     uint32_t id = ground_truth_ids[i];
                    //     auto const &pool = nhoods[id].pool;
                    //     auto truth = ground_truth_knng.at(i);
                    //     float count_right=0;
                    //     unsigned neighbornumber=params.K;
                    //     for(unsigned j=0 ;j<neighbornumber;j++)
                    //     {
                    //         auto iter=find(truth.begin(), truth.begin()+neighbornumber, pool[j].id);
                    //         if (distance(truth.begin(),iter)<neighbornumber) {
                    //             count_right++;
                    //         }
                    //     }
                    //     sum_recall += (count_right / neighbornumber);
                    // }
                    float myrecall=computeOverallRecall(ground_truth_knng,ground_truth_ids);
                    cout<<myrecall<<endl;
                    info.delta = mean(delta);
                    info.recall = mean(recall);
                    info.M = mean(M);
                    auto times = timer.elapsed();
                    info.time = times.wall / 1e9;
                    if (verbosity > 0) {
                        all_cost[it]=info.cost;
                        all_recall[it]=myrecall;
                        all_time[it]=info.time;
                        all_peakMemory[it]=getPeakMemoryUsage();
                        cerr << "iteration: " << info.iterations
                             << " recall: " << info.recall
                             << " accuracy: " << info.accuracy
                             << " cost: " << info.cost
                             << " M: " << info.M
                             << " delta: " << info.delta
                             << " time: " << info.time
                             << " one-recall: " << mean(one_recall)
                             << " one-ratio: " << mean(one_approx) / mean(one_exact)
                             << endl;
                    }
                }
                if (info.delta <= params.delta) {
                    info.stop_condition = IndexInfo::DELTA;
                    break;
                }
                if (info.recall >= params.recall) {
                    info.stop_condition = IndexInfo::RECALL;
                    break;
                }
                // if(info.iterations>=params.iterations)
                // {
                //     break;
                // }
                auto times3 = timer.elapsed();
                update();
                auto times4 = timer.elapsed();
                float update_time=times4.wall / 1e9-times3.wall / 1e9;
                all_update[it]=update_time;

                /*
                std::cout<<"iter:"<<it+1<<"    update time(s): "<<times4.wall / 1e9-times3.wall / 1e9<<std::endl;
                std::cout<<"iter:"<<it+1<<"  end time(s): "<<times4.wall / 1e9<<std::endl;
                */
            }
            string path=truth_path;
            size_t lastSlashPos = path.find_last_of('/');
            string data = path.substr(lastSlashPos + 1);
            write_info(all_cost,all_recall,all_time,all_peakMemory,all_join,all_update,FixedMemory,data);
            if (params.reverse) {
                reverse(params.reverse);
            }
            if (params.prune) {
                prune(o, params.prune);
            }
            if (pinfo) {
                *pinfo = info;
            }
        }

    };

    void KGraphImpl::build (IndexOracle const &oracle, IndexParams const &param, IndexInfo *info,char const *path,char const *truth_path) {
        KGraphConstructor con(oracle, param, info, truth_path);
        if(path!=nullptr)
        {
            con.save_knng(path);
        }
        else
        {
            //std::cout<<"path is nullptr!";
        }
        //M.swap(con.M);
        //graph.swap(con.graph);
        //std::swap(no_dist, con.no_dist);
    }

    void KGraphImpl::save_groundtruth(IndexOracle const &oracle,char const *path){
        vector<Control> controls;
        GenerateControl(oracle, 500, 100, &controls);
        vector<unsigned> ids;
        ids.resize(500);
        vector<vector<unsigned>> knng;
        knng.resize(500);
        for(unsigned i=0;i<500;i++)
        {
            ids[i]=(controls[i].id);
            auto &knn=knng[i];
            knn.resize(100);
            for(unsigned j=0;j<100;j++)
            {
                auto neighbors=controls[i].neighbors;
                knn[j]=neighbors[j].id;
            }
        }
        for(int i=0;i<500;i++)
        {
            cout<<ids[i]<<" ";
        }
        std::ofstream ofs(path, std::ios::out | std::ios::binary);
        cout << "Saving to " << path << endl;
        const uint32_t num_points = ids.size();
        ofs.write(reinterpret_cast<char const *>(&num_points), sizeof(uint32_t));
        const int K = knng.front().size();
        for (unsigned i = 0; i < ids.size(); ++i) {
            auto const &knn = knng[i];
            ofs.write(reinterpret_cast<char const *>(&ids[i]), sizeof(uint32_t));
            ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(uint32_t));
        }
        ofs.close();
    }

    KGraph *KGraph::create () {
        return new KGraphImpl;
    }

    char const* KGraph::version () {
        return kgraph_version;
    }
}

