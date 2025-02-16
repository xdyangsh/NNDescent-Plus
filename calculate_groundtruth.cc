#include <sys/time.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "assert.h"

using namespace std;
#define _INT_MAX 2147483640

float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs, uint32_t &dim) {
  float ans = 0.0;
  int lensDim = dim;

  for (int i = 0; i < lensDim; ++i) {
    ans += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
  }
  return ans;
}

vector<uint32_t> GreedyNearest(const vector<vector<float>> &dpts,
                               const vector<float> query,
                               const int k_smallest,
                               uint32_t &dim) {
  std::priority_queue<std::pair<float, uint32_t>> top_candidates;
  float lower_bound = _INT_MAX;
  for (size_t i = 0; i < dpts.size(); i++) {
    float dist = EuclideanDistance(query, dpts[i],dim);
    if (top_candidates.size() < k_smallest || dist < lower_bound) {
      top_candidates.push(std::make_pair(dist, i));
      if (top_candidates.size() > k_smallest) {
        top_candidates.pop();
      }

      lower_bound = top_candidates.top().first;
    }
  }
  vector<uint32_t> res;
  while (!top_candidates.empty()) {
    res.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  std::reverse(res.begin(), res.end());
  return res;
}


inline void ReadLSHKIT(const std::string &file_path,
                    std::vector<std::vector<float>> &data,uint32_t &dim) {
  std::cout << "Reading Data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
  assert(ifs.is_open());
  uint32_t N;  // num of points
  uint32_t num_dimensions; // dim of point

  ifs.read((char *)&N, sizeof(uint32_t));
  ifs.read((char *)&N, sizeof(uint32_t));
  ifs.read((char *)&num_dimensions,sizeof(uint32_t));
  dim=num_dimensions;
  data.resize(N);
  std::cout << "# of points: " << N << std::endl;
  std::vector<float> buff(num_dimensions);
  int counter = 0;
  while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
    std::vector<float> row(num_dimensions);
    for (int d = 0; d < num_dimensions; d++) {
      row[d] = static_cast<float>(buff[d]);
    }
    data[counter++] = std::move(row);
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}

// first N, then N * (1 + 100)uint32_t
inline void SaveSampleGroundtruth(
    const std::vector<std::vector<uint32_t>> &knng,
    const std::vector<uint32_t> &ids, const string &path) {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  //std::ofstream ofs(path, std::ios::out);
  cout << "Saving to " << path << endl;

  const uint32_t num_points = ids.size();  // N
  ofs.write(reinterpret_cast<char const *>(&num_points), sizeof(uint32_t));
  const int K = knng.front().size();
  for (unsigned i = 0; i < ids.size(); ++i) {
    auto const &knn = knng[i];
    ofs.write(reinterpret_cast<char const *>(&ids[i]), sizeof(uint32_t));
    ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(uint32_t));
  }
  ofs.close();
}

void CalculateGroundtruthSample(const vector<vector<float>> &nodes,
                                const vector<uint32_t> &ids,
                                const string &gt_path,
                                uint32_t &dim,
                                int target_K) {
  vector<vector<uint32_t>> groundtruth;
  groundtruth.resize(ids.size());
#pragma omp parallel for
  for (unsigned n = 0; n < ids.size(); n++) {
    groundtruth[n] = GreedyNearest(nodes, nodes[ids[n]], target_K+1,dim);
    // pop itself
    groundtruth[n].erase(groundtruth[n].begin());
  }
  assert(groundtruth.size() == ids.size());
  SaveSampleGroundtruth(groundtruth, ids, gt_path);
}

void logTime(timeval &begin, timeval &end, const string &log) {
  gettimeofday(&end, NULL);
  fprintf(stdout, ("# " + log + ": %.7fs\n").c_str(),
          end.tv_sec - begin.tv_sec +
              (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC);
};

int main(int argc, char **argv) {
  string source_path;
  int target_size;
  int target_K;
  string target_path;

  for (int i = 0; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-source") source_path = string(argv[i + 1]);
    if (arg == "-size") target_size = atoi(argv[i + 1]);
    if (arg == "-K") target_K = atoi(argv[i + 1]);
    if (arg == "-target") target_path = string(argv[i + 1]);
  }

  vector<vector<float>> nodes;
  uint32_t dim=0;
  ReadLSHKIT(source_path, nodes,dim);
  timeval tt1, tt2;

  std::random_device rd;
  std::mt19937 gen(rd());  // Mersenne twister MT19937
  vector<uint32_t> idxes(nodes.size());
  iota(idxes.begin(), idxes.end(), 0);
  shuffle(idxes.begin(), idxes.end(), gen);
  idxes.resize(target_size);
  sort(idxes.begin(),idxes.end());
  std::cout<<"num of gt: "<<idxes.size()<<std::endl;
  for(int i=0;i<idxes.size();i++)
  {
    std::cout<<idxes[i]<<"  ";
  }
  std::cout<<std::endl;

  gettimeofday(&tt1, NULL);
  CalculateGroundtruthSample(nodes, idxes, target_path,dim,target_K);
  gettimeofday(&tt2, NULL);
  logTime(tt1, tt2, "Cost Time");
  return 0;
}

//  g++ -O3 -o gt calculate_groundtruth.cc -fopenmp 
// ./gt -source /path/to/dataset.lshkit -target dataset_gt.ibin -K 500 -size 10000