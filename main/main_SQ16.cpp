#include <vector>
#include "VectorOracle.h"
#include "kgraph.h"
#include <stdlib.h>
#include <chrono>
using namespace std;
using namespace kgraph;

int main(int argc, char **argv) {
    double read_time = 0;
    std::chrono::system_clock::time_point start, end;

    string source_path = "/home/yangshuo/Codes/datasets/sift_base.lshkit";
    string truth_path = "/home/yangshuo/Codes/datasets/groundtruth/sift_base_500_100.ibin";
    char *ptr;
    Matrix<unsigned short> data;
    KGraph::IndexParams params;
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-source") source_path = string(argv[i + 1]);
        if (arg == "-truth") truth_path = string(argv[i + 1]);
        if (arg == "-L") params.L = (unsigned)strtoul(argv[i + 1],&ptr,10);
        if (arg == "-S") params.S = (unsigned)strtoul(argv[i + 1],&ptr,10);
        if (arg == "-R") params.R = (unsigned)strtoul(argv[i + 1],&ptr,10);
        if (arg == "-iter") params.iterations = (unsigned)strtoul(argv[i + 1],&ptr,10);
        if (arg == "-K") params.K = (unsigned)strtoul(argv[i + 1],&ptr,10);
    }
    std::cout<<"L="<<params.L<<"  S="<<params.S<<"  R="<<params.R<<"  iter="<<params.iterations<<"  K="<<params.K<<std::endl;
    start = std::chrono::system_clock::now();
    data.load_lshkit_unsigned_short(source_path);
    
    end = std::chrono::system_clock::now();
	read_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " read_time: " << read_time << "[msec]\n\n";
    MatrixOracle<unsigned short> oracle(data);
    oracle.test();
    KGraph::IndexInfo info;
    KGraph *kgraph = KGraph::create(); //(oracle, params, &info);
    kgraph->build(oracle, params, &info,nullptr,truth_path.c_str());
    delete kgraph;
    return 0;
}