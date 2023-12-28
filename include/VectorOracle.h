using namespace std;
#include "Euclidean.h"
#include <cmath>
#include <cstring>
#include <malloc.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <boost/assert.hpp>
#include <iostream>
#include <cfloat>
#include "kgraph.h"
#include <algorithm>
#include <queue>

#ifdef __AVX2__
#define KGRAPH_MATRIX_ALIGN 32
#else
#ifdef __SSE__
#define KGRAPH_MATRIX_ALIGN 16
#else
#define KGRAPH_MATRIX_ALIGN 16
#endif
#endif

#define BIT_NUM 14
template <typename T, unsigned A = KGRAPH_MATRIX_ALIGN>
class Matrix {
    unsigned col;
    unsigned row;
    size_t stride;
    char *data;
    float *length;

    void reset (unsigned r, unsigned c) {
        row = r;
        col = c;
        stride = (sizeof(T) * c + A - 1) / A * A;
        cout<<"stride: "<<stride<<endl;
        /*
        data.resize(row * stride);
        */
        if (data) free(data);
        data = (char *)memalign(A, row * stride); // SSE instruction needs data to be aligned
        if (!data) throw runtime_error("memalign");
        if(length) free(length);
        length= (float *)memalign(A,(sizeof(float) * c + A - 1) / A * A);
        memset(length, 0, (sizeof(float) * c + A - 1) / A * A);
    }
public:
    Matrix (): col(0), row(0), stride(0), data(0),length(0) {}
    Matrix (unsigned r, unsigned c): data(0) {
        reset(r, c);
    }
    ~Matrix () {
        if (data) free(data);
    }
    unsigned size () const {
        return row;
    }
    unsigned dim () const {
        return col;
    }
    size_t step () const {
        return stride;
    }
    float* getlen() const {
        return length;
    }
    void resize (unsigned r, unsigned c) {
        reset(r, c);
    }
    T const *operator [] (unsigned i) const {
        return reinterpret_cast<T const *>(&data[stride * i]);
    }
    T *operator [] (unsigned i) {
        return reinterpret_cast<T *>(&data[stride * i]);
    }
    void zero () {
        memset(data, 0, row * stride);
    }
    
    void load (const std::string &file_path, unsigned dim, unsigned skip = 0, unsigned gap = 0) {
        double read_time = 0;
        std::chrono::system_clock::time_point start, end;
        start = std::chrono::system_clock::now();
        std::cout << "Reading Data: " << file_path << std::endl;
        std::ifstream ifs;
        ifs.open(file_path, std::ios::binary);
        assert(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        size -= skip;
        unsigned line = sizeof(float) * dim + gap;
        unsigned N =  size / line;
        reset(N, dim);
        zero();
        ifs.seekg(skip, std::ios::beg);
        
        for (unsigned i = 0; i < N; ++i) {
            ifs.read(&data[stride * i], sizeof(float) * dim);
            ifs.seekg(gap, std::ios::cur);
        }
        //ifs.read(&data[0],sizeof(float)*dim*N);
        ifs.close();
        std::cout << "Finish Reading Data" << endl;
        end = std::chrono::system_clock::now();
	    read_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "float read time: " << read_time << "[msec]\n\n";
    }

    void load_lshkit (std::string const &path) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof(header));
        is.close();
        unsigned D = header[2];
        unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
        unsigned gap = 0;
        load(path, D, skip, gap);
    }
    void load_unsigned_short(const std::string &file_path, unsigned dim, unsigned skip = 0, unsigned gap = 0)
    {
        double read_time = 0;
        std::chrono::system_clock::time_point start, end;
        start = std::chrono::system_clock::now();
        std::cout << "Reading Data: " << file_path << std::endl;
        std::ifstream ifs;
        ifs.open(file_path, std::ios::binary);
        assert(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        size -= skip;
        unsigned line = sizeof(float) * dim + gap;
        unsigned N =  size / line;
        reset(N, dim);
        zero();
        ifs.seekg(skip, ios::beg);
        std::vector<float> buff(dim);
        float max_val=FLT_MIN;
        float min_val=FLT_MAX;
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<float> row(dim);
            for (int d = 0; d < dim; d++) {
                row[d] = static_cast<float>(buff[d]);
                if(row[d]<min_val)
                {
                    min_val=row[d];
                }
                if(row[d]>max_val)
                {
                    max_val=row[d];
                }
            }
            ifs.seekg(gap, std::ios::cur);
        }
        unsigned int len=0;
       /*if(dim>=64)
        {
            len = sqrt(4294967295/(dim/4));
        }
        else
        {
            len = sqrt(4294967295/dim);
        }*/
        len=65536;
        cout<<"len:"<<len<<endl;
        std::cout<<len<<std::endl;
        ifs.seekg(skip, ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<unsigned short> row(dim);
            for (int d = 0; d < dim; d++) {
                row[d] = static_cast<unsigned short>(static_cast<int>((static_cast<float>(buff[d])-min_val)/(max_val-min_val)*(len-1)));
            }
            memcpy(&data[stride * i],&row[0],sizeof(T) * dim);
            ifs.seekg(gap, std::ios::cur);
        }
        ifs.close();
        std::cout << "Finish Reading Data" << endl;
        end = std::chrono::system_clock::now();
	    read_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "short read time: " << read_time << "[msec]\n\n";
    }

    void load_lshkit_unsigned_short (std::string const &path) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof(header));
        is.close();
        unsigned D = header[2];
        unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
        unsigned gap = 0;
        load_unsigned_short(path, D, skip, gap);
    }
    void load_unsigned_char(const std::string &file_path, unsigned dim, unsigned skip = 0, unsigned gap = 0)
    {
        std::cout << "Reading Data: " << file_path << std::endl;
        std::ifstream ifs;
        ifs.open(file_path, std::ios::binary);
        assert(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        size -= skip;
        unsigned line = sizeof(float) * dim + gap;
        unsigned N =  size / line;
        reset(N, dim);
        zero();
        ifs.seekg(skip, ios::beg);
        std::vector<float> buff(dim);
        float max_val=FLT_MIN;
        float min_val=FLT_MAX;
        priority_queue<float, vector<float>,less<float>> bigtop;
        priority_queue<float, vector<float>,greater<float>> smalltop;
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<float> row(dim);
            for (int d = 0; d < dim; d++) {
                row[d] = static_cast<float>(buff[d]);
                if(bigtop.size()<dim*N*0.0001)
                {
                    bigtop.push(row[d]);
                    smalltop.push(row[d]);
                }
                else
                {
                    if(row[d]<bigtop.top())
                    {
                        bigtop.pop();
                        bigtop.push(row[d]);
                    }
                    if(row[d]>smalltop.top())
                    {
                        smalltop.pop();
                        smalltop.push(row[d]);
                    }
                }
                if(row[d]<min_val)
                {
                    min_val=row[d];
                }
                if(row[d]>max_val)
                {
                    max_val=row[d];
                }
            }
            ifs.seekg(gap, std::ios::cur);
        }
        cout<<min_val<<endl;
        cout<<max_val<<endl;
        unsigned short len=0;
        
       /*if(dim>=64)
        {
            len = sqrt(4294967295/(dim/4));
        }
        else
        {
            len = sqrt(4294967295/dim);
        }*/
        min_val=bigtop.top();
        max_val=smalltop.top();
        cout<<min_val<<endl;
        cout<<max_val<<endl;
        len=256;
        cout<<"len:"<<len<<endl;
        std::cout<<len<<std::endl;
        ifs.seekg(skip, ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<unsigned char> row(dim);
            for (int d = 0; d < dim; d++) {
                if(static_cast<float>(buff[d])>max_val)
                {
                    row[d]=len-1;
                }
                else if(static_cast<float>(buff[d])<min_val)
                {
                    row[d]=0;
                }
                else{
                row[d] = static_cast<unsigned char>(static_cast<int>((static_cast<float>(buff[d])-min_val)/(max_val-min_val)*(len-1)));
                }
            }
            memcpy(&data[stride * i],&row[0],sizeof(T) * dim);
            ifs.seekg(gap, std::ios::cur);
        }
        ifs.close();
        std::cout << "Finish Reading Data" << endl;
    }

    void load_lshkit_unsigned_char (std::string const &path) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof(header));
        is.close();
        unsigned D = header[2];
        unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
        unsigned gap = 0;
        load_unsigned_char(path, D, skip, gap);
    }

    void load_unsigned_short_group(const std::string &file_path, unsigned dim, unsigned skip = 0, unsigned gap = 0)
    {
        std::cout << "Reading Data: " << file_path << std::endl;
        std::ifstream ifs;
        ifs.open(file_path, std::ios::binary);
        assert(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        size -= skip;
        unsigned line = sizeof(float) * dim + gap;
        unsigned N =  size / line;
        reset(N, dim);
        zero();
        ifs.seekg(skip, ios::beg);
        std::vector<float> buff(dim);
        vector<float> max_vec(ceil(dim/group_num),FLT_MIN);
        vector<float> min_vec(ceil(dim/group_num),FLT_MAX);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<float> row(dim);
            for (int d = 0; d < dim; d++) {
                int index=floor(d/group_num);
                row[d] = static_cast<float>(buff[d]);
                if(row[d]<min_vec[index])
                {
                    min_vec[index]=row[d];
                }
                if(row[d]>max_vec[index])
                {
                    max_vec[index]=row[d];
                }
            }
            ifs.seekg(gap, std::ios::cur);
        }
        unsigned int len=0;
       /*if(dim>=64)
        {
            len = sqrt(4294967295/(dim/1));
        }
        else
        {
            len = sqrt(4294967295/dim);
        }*/
        len=65536;
        cout<<"len:"<<len<<endl;
        std::cout<<len<<std::endl;
        ifs.seekg(skip, ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<unsigned short> row(dim);
            for (int d = 0; d < dim; d++) {
                row[d] = static_cast<unsigned short>(static_cast<int>((static_cast<float>(buff[d])-min_vec[floor(d/group_num)])/(max_vec[floor(d/group_num)]-min_vec[floor(d/group_num)])*(len-1)));
            }
            memcpy(&data[stride * i],&row[0],sizeof(T) * dim);
            ifs.seekg(gap, std::ios::cur);
        }
        ifs.close();
        for(int d=0;d<dim;d++)
        {
            length[d]=(max_vec[floor(d/group_num)]-min_vec[floor(d/group_num)])/len;
        }
        std::cout << "Finish Reading Data" << endl;
    }

    void load_lshkit_unsigned_short_group (std::string const &path) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof(header));
        is.close();
        unsigned D = header[2];
        unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
        unsigned gap = 0;
        load_unsigned_short_group(path, D, skip, gap);
    }
    
    void load_unsigned_char_group(const std::string &file_path, unsigned dim, unsigned skip = 0, unsigned gap = 0)
    {
        std::cout << "Reading Data: " << file_path << std::endl;
        std::ifstream ifs;
        ifs.open(file_path, std::ios::binary);
        assert(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        size -= skip;
        unsigned line = sizeof(float) * dim + gap;
        unsigned N =  size / line;
        reset(N, dim);
        zero();
        ifs.seekg(skip, ios::beg);
        std::vector<float> buff(dim);
        //vector<vector<float>> all_ele(ceil(dim/group_num));
        vector<priority_queue<float, vector<float>,less<float>>> bigtop(ceil(dim/group_num));
        vector<priority_queue<float, vector<float>,greater<float>>> smalltop(ceil(dim/group_num));
        vector<float> max_vec(ceil(dim/group_num),FLT_MIN);
        vector<float> min_vec(ceil(dim/group_num),FLT_MAX);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<float> row(dim);
            for (int d = 0; d < dim; d++) {
                int index=floor(d/group_num);
                row[d] = static_cast<float>(buff[d]);
                if(bigtop[index].size()<N*group_num*0.0001)
                {
                    bigtop[index].push(row[d]);
                    smalltop[index].push(row[d]);
                }
                else
                {
                    if(row[d]<bigtop[index].top())
                    {
                        bigtop[index].pop();
                        bigtop[index].push(row[d]);
                    }
                    if(row[d]>smalltop[index].top())
                    {
                        smalltop[index].pop();
                        smalltop[index].push(row[d]);
                    }
                }
                //all_ele[index].push_back(row[d]);
                if(row[d]<min_vec[index])
                {
                    min_vec[index]=row[d];
                }
                if(row[d]>max_vec[index])
                {
                    max_vec[index]=row[d];
                }
            }
            ifs.seekg(gap, std::ios::cur);
        }
        unsigned short len=0;
        for(int i=0;i<max_vec.size();i++)
        {
            min_vec[i]=bigtop[i].top();
            max_vec[i]=smalltop[i].top();
        }
       /*if(dim>=64)
        {
            len = sqrt(4294967295/(dim/1));
        }
        else
        {
            len = sqrt(4294967295/dim);
        }*/
        len=256;
        cout<<"len:"<<len<<endl;
        std::cout<<len<<std::endl;
        ifs.seekg(skip, ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<unsigned char> row(dim);
            for (int d = 0; d < dim; d++) {
                if(static_cast<float>(buff[d])<min_vec[floor(d/group_num)])
                {
                    row[d]=0;
                }
                else if(static_cast<float>(buff[d])>max_vec[floor(d/group_num)])
                {
                    row[d]=len-1;
                }
                else
                {
                row[d] = static_cast<unsigned char>(static_cast<int>((static_cast<float>(buff[d])-min_vec[floor(d/group_num)])/(max_vec[floor(d/group_num)]-min_vec[floor(d/group_num)])*(len-1)));
                }
            }
            memcpy(&data[stride * i],&row[0],sizeof(T) * dim);
            ifs.seekg(gap, std::ios::cur);
        }
        ifs.close();
        for(int d=0;d<dim;d++)
        {
            float temp=(max_vec[floor(d/group_num)]-min_vec[floor(d/group_num)])/len;
            length[d]=temp*temp;
        }
        std::cout << "Finish Reading Data" << endl;
    }

    void load_lshkit_unsigned_char_group (std::string const &path) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof(header));
        is.close();
        unsigned D = header[2];
        unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
        unsigned gap = 0;
        load_unsigned_char_group(path, D, skip, gap);
    }

    void load_unsigned_char_groupmix(const std::string &file_path, unsigned dim, unsigned skip = 0, unsigned gap = 0)
    {
        std::cout << "Reading Data: " << file_path << std::endl;
        std::ifstream ifs;
        ifs.open(file_path, std::ios::binary);
        assert(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        size -= skip;
        unsigned line = sizeof(float) * dim + gap;
        unsigned N =  size / line;
        reset(N, dim);
        zero();
        ifs.seekg(skip, ios::beg);
        std::vector<float> buff(dim);
        //vector<vector<float>> all_ele(ceil(dim/group_num));
        vector<priority_queue<float, vector<float>,less<float>>> bigtop(group_num);
        vector<priority_queue<float, vector<float>,greater<float>>> smalltop(group_num);
        vector<float> max_vec(group_num,FLT_MIN);
        vector<float> min_vec(group_num,FLT_MAX);
        unsigned numInOneGroup=ceil(dim/group_num);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<float> row(dim);
            for (int d = 0; d < dim; d++) {
                int index=floor(d/numInOneGroup);
                row[d] = static_cast<float>(buff[d]);
                if((index<(group_num-1)&&bigtop[index].size()<N*numInOneGroup*0.0001)||
                (index==(group_num-1)&&bigtop[index].size()<N*(dim-numInOneGroup*(group_num-1))*0.0001))
                {
                    bigtop[index].push(row[d]);
                    smalltop[index].push(row[d]);
                }
                else
                {
                    if(row[d]<bigtop[index].top())
                    {
                        bigtop[index].pop();
                        bigtop[index].push(row[d]);
                    }
                    if(row[d]>smalltop[index].top())
                    {
                        smalltop[index].pop();
                        smalltop[index].push(row[d]);
                    }
                }
                //all_ele[index].push_back(row[d]);
                if(row[d]<min_vec[index])
                {
                    min_vec[index]=row[d];
                }
                if(row[d]>max_vec[index])
                {
                    max_vec[index]=row[d];
                }
            }
            ifs.seekg(gap, std::ios::cur);
        }
        unsigned short len=0;
        for(int i=0;i<max_vec.size();i++)
        {
            min_vec[i]=bigtop[i].top();
            max_vec[i]=smalltop[i].top();
        }
       /*if(dim>=64)
        {
            len = sqrt(4294967295/(dim/1));
        }
        else
        {
            len = sqrt(4294967295/dim);
        }*/
        len=256;
        cout<<"len:"<<len<<endl;
        std::cout<<len<<std::endl;
        ifs.seekg(skip, ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<unsigned char> row(stride,0);
            for (int d = 0; d < dim; d++) {
                if(static_cast<float>(buff[d])<min_vec[floor(d/numInOneGroup)])
                {
                    row[(d%numInOneGroup)*group_num+d/numInOneGroup]=0;
                }
                else if(static_cast<float>(buff[d])>max_vec[floor(d/numInOneGroup)])
                {
                    row[(d%numInOneGroup)*group_num+d/numInOneGroup]=len-1;
                }
                else
                {
                row[(d%numInOneGroup)*group_num+d/numInOneGroup] = static_cast<unsigned char>(static_cast<int>((static_cast<float>(buff[d])-min_vec[floor(d/numInOneGroup)])/(max_vec[floor(d/numInOneGroup)]-min_vec[floor(d/numInOneGroup)])*(len-1)));
                }
            }
            memcpy(&data[stride * i],&row[0],sizeof(T) * stride);
            ifs.seekg(gap, std::ios::cur);
        }
        ifs.close();
        for(int d=0;d<group_num;d++)
        {
            float temp=(max_vec[d]-min_vec[d])/len;
            length[d]=temp*temp;
        }
        std::cout << "Finish Reading Data" << endl;
    }

    void load_lshkit_unsigned_char_groupmix (std::string const &path) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof(header));
        is.close();
        unsigned D = header[2];
        unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
        unsigned gap = 0;
        load_unsigned_char_groupmix(path, D, skip, gap);
    }

    void load_unsigned_char_group_select(const std::string &file_path, unsigned dim, unsigned skip = 0, unsigned gap = 0)
    {
        std::cout << "Reading Data: " << file_path << std::endl;
        std::ifstream ifs;
        ifs.open(file_path, std::ios::binary);
        assert(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        size -= skip;
        unsigned line = sizeof(float) * dim + gap;
        unsigned N =  size / line;
        reset(N, dim);
        zero();
        ifs.seekg(skip, ios::beg);
        std::vector<float> buff(dim);
        //vector<vector<float>> all_ele(ceil(dim/group_num));
        vector<float> max_vec(dim,FLT_MIN);
        vector<float> min_vec(dim,FLT_MAX);
        vector<float> delta_vec(dim,0);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<float> row(dim);
            for (int d = 0; d < dim; d++) {
                //all_ele[index].push_back(row[d]);
                row[d] = static_cast<float>(buff[d]);
                if(row[d]<min_vec[d])
                {
                    min_vec[d]=row[d];
                    
                }
                if(row[d]>max_vec[d])
                {
                    max_vec[d]=row[d];
                    
                }
            }
            ifs.seekg(gap, std::ios::cur);
        }
        for(int i=0;i<dim;i++)
        {
            delta_vec[i]=max_vec[i]-min_vec[i];
        }
        vector<float> select_max_vec(ceil(dim/group_num),FLT_MIN);
        vector<float> select_min_vec(ceil(dim/group_num),FLT_MAX);
        vector<float> select_delta_vec(ceil(dim/group_num),0);
        vector<int> index_vec;
        for(int i=0;i<dim;i++)
        {
            if(i%group_num==0)
            {
                int temp=0;
                temp=min_element(delta_vec.begin(),delta_vec.end())-delta_vec.begin();
                index_vec.push_back(temp);
                select_min_vec[floor(i/group_num)]=min_vec[temp];
                select_max_vec[floor(i/group_num)]=max_vec[temp];
                select_delta_vec[floor(i/group_num)]=max_vec[temp]-min_vec[temp];
                delta_vec[temp]=FLT_MAX;
            }
            else
            {
                float max_val=select_max_vec[floor(i/group_num)];
                float min_val=select_min_vec[floor(i/group_num)];
                float delta = FLT_MAX;
                int index=0;
                int sum=0;
                for(int j=0;j<dim;j++)
                {
                    float temp_max = std::max(max_val,max_vec[j]);
                    float temp_min= std::min(min_val,min_vec[j]);
                    float temp_delta = temp_max-temp_min;
                    if (std::find(index_vec.begin(), index_vec.end(), j) == index_vec.end())
                    {
                        if(temp_delta<delta)
                        {
                            delta=temp_delta;
                            index=j;
                        }
                    }
                }
                index_vec.push_back(index);
                select_min_vec[floor(i/group_num)]=min_vec[index];
                select_max_vec[floor(i/group_num)]=max_vec[index];
                select_delta_vec[floor(i/group_num)]=max_vec[index]-min_vec[index];
                delta_vec[index]=FLT_MAX;
            }
        }
        cout<<endl;
        unsigned short len=0;
       /*if(dim>=64)
        {
            len = sqrt(4294967295/(dim/1));
        }
        else
        {
            len = sqrt(4294967295/dim);
        }*/
        len=256;
        cout<<"len:"<<len<<endl;
        std::cout<<len<<std::endl;
        ifs.seekg(skip, ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<unsigned char> row(dim);
            for (int d = 0; d < dim; d++) {
                row[d] = static_cast<unsigned char>(static_cast<int>((static_cast<float>(buff[index_vec[d]])-select_min_vec[floor(d/group_num)])/(select_max_vec[floor(d/group_num)]-select_min_vec[floor(d/group_num)])*(len-1)));
            }
            memcpy(&data[stride * i],&row[0],sizeof(T) * dim);
            ifs.seekg(gap, std::ios::cur);
        }
        for(int d=0;d<dim;d++)
        {
            length[d]=(select_max_vec[floor(d/group_num)]-select_min_vec[floor(d/group_num)])/len;
        }
        ifs.close();
        std::cout << "Finish Reading Data" << endl;
    }

    void load_lshkit_unsigned_char_group_select (std::string const &path) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof(header));
        is.close();
        unsigned D = header[2];
        unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
        unsigned gap = 0;
        load_unsigned_char_group_select(path, D, skip, gap);
    }

    void load_bit(const std::string &file_path, unsigned dim, unsigned skip = 0, unsigned gap = 0)
    {
        std::cout << "Reading Data: " << file_path << std::endl;
        std::ifstream ifs;
        ifs.open(file_path, std::ios::binary);
        assert(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        size_t size = ifs.tellg();
        size -= skip;
        unsigned line = sizeof(float) * dim + gap;
        unsigned N =  size / line;
        {
            row = N;
            col = dim;
            stride = floor((float(BIT_NUM)/(sizeof(T)*8) * col + A - 1) / A) * A;
            cout<<"stride: "<<stride<<endl;
            /*
            data.resize(row * stride);
            */
            if (data) free(data);
            data = (char *)memalign(A, row * stride); // SSE instruction needs data to be aligned
            if (!data) throw runtime_error("memalign");
            memset(data, 0, row * stride);
            if(length) free(length);
            length= (float *)memalign(A,(sizeof(float) * col + A - 1) / A * A);
            memset(length, 0, (sizeof(float) * col + A - 1) / A * A);
            zero();
        }
        ifs.seekg(skip, ios::beg);
        std::vector<float> buff(dim);
        float max_val=FLT_MIN;
        float min_val=FLT_MAX;
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<float> row(dim);
            for (int d = 0; d < dim; d++) {
                row[d] = static_cast<float>(buff[d]);
                if(row[d]<min_val)
                {
                    min_val=row[d];
                }
                if(row[d]>max_val)
                {
                    max_val=row[d];
                }
            }
            ifs.seekg(gap, std::ios::cur);
        }
        unsigned int len=0;
        len=65536;
        cout<<"len:"<<len<<endl;
        std::cout<<len<<std::endl;
        ifs.seekg(skip, ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            ifs.read((char *)buff.data(), dim * sizeof(float));
            std::vector<unsigned short> row(dim);
            std::vector<unsigned char> temp(stride,0x00);
            int currentLen=8;//当前unsigned char可用长度
            int current=stride-1; //当前temp下标
            for (int d = 0; d < dim; d++) {
                row[d] = static_cast<unsigned short>(static_cast<int>((static_cast<float>(buff[d])-min_val)/(max_val-min_val)*(len-1)));
                if(i==0)
                {
                    cout<<row[d]<<" ";
                }
                int current_bit=BIT_NUM;//当前还需编码长度
                while(current_bit>0)
                {
                    if(currentLen>0&&current_bit>=currentLen)
                    {    
                        unsigned char mask = (1 << currentLen) - 1;
                        temp[current]=(temp[current]) | ((row[d] & mask) << (8 - currentLen));
                        row[d]=row[d]>>currentLen;
                        current_bit-=currentLen;
                        currentLen=0;
                    }
                    else if(currentLen>0&&current_bit<currentLen)
                    {
                        unsigned char mask = (1 << current_bit) - 1;
                        temp[current]=(temp[current]) | ((row[d] & mask) << (8 - currentLen));
                        row[d]=row[d]>>current_bit;
                        currentLen-=current_bit;
                        current_bit=0;
                    }
                    else
                    {
                        current--;
                        currentLen=8;
                    }
                }
            }
            
            if(i==0){
                cout<<endl;
            //     for(int i=0;i<stride;i++)
            // {
            //     cout<<(unsigned int)temp[i]<<"   ";
            // }
            // cout<<endl;
            }
            
            memcpy(&data[stride * i],&temp[0],sizeof(T) * stride);
            ifs.seekg(gap, std::ios::cur);
        }
        ifs.close();
        // decoding((unsigned char*)(data),dim,stride);
        std::cout << "Finish Reading Data" << endl;
    }
    void load_lshkit_bit (std::string const &path) {
        static const unsigned LSHKIT_HEADER = 3;
        std::ifstream is(path.c_str(), std::ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof(header));
        is.close();
        unsigned D = header[2];
        unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
        unsigned gap = 0;
        load_bit(path, D, skip, gap);
    }
};

template <typename DATA_TYPE, unsigned A = KGRAPH_MATRIX_ALIGN>
class MatrixProxy {
    unsigned rows;
    unsigned cols;      // # elements, not bytes, in a row, 
    size_t stride;    // # bytes in a row, >= cols * sizeof(element)
    uint8_t const *data;
    float const *length;
public:
    MatrixProxy (Matrix<DATA_TYPE> const &m)
        : rows(m.size()), cols(m.dim()), stride(m.step()), data(reinterpret_cast<uint8_t const *>(m[0])),length(m.getlen()) {
    }
    unsigned size () const {
        return rows;
    }
    unsigned dim () const {
        return cols;
    }
    size_t step() const {
        return stride;
    }
    float const* getlen() const
    {
        return length;
    }
    DATA_TYPE const *operator [] (unsigned i) const {
        return reinterpret_cast<DATA_TYPE const *>(data + stride * i);
    }
    DATA_TYPE *operator [] (unsigned i) {
        return const_cast<DATA_TYPE *>(reinterpret_cast<DATA_TYPE const *>(data + stride * i));
    }
};

template <typename DATA_TYPE>
class MatrixOracle:public kgraph::IndexOracle{
    MatrixProxy<DATA_TYPE> proxy;
public:
    template <typename MATRIX_TYPE>
    MatrixOracle (MATRIX_TYPE const &m): proxy(m) {
    }
    virtual unsigned size() const{
        return proxy.size();
    }
    virtual unsigned dim() const{
        return proxy.dim();
    }
    virtual float const* length() const{
        return proxy.getlen();
    }
    virtual dist_type operator()(unsigned i,unsigned j) const{
#ifdef __AVX2__
    #ifdef SHORT
        //cout<<"EuclideanDistanceShortAVX(proxy[i],proxy[j],proxy.dim())"<<endl;
        return EuclideanDistanceShortAVX(proxy[i],proxy[j],proxy.dim());
    #else
    #ifdef CHAR
        #ifdef GROUP
            //cout<<"EuclideanDistanceGroupAVX(proxy[i],proxy[j],proxy.dim(),proxy.getlen())"<<endl;
            return EuclideanDistanceGroupAVX(proxy[i],proxy[j],proxy.dim(),proxy.getlen());
        #else
        #ifdef ALL
            //cout<<"EuclideanDistanceAVX(proxy[i],proxy[j],proxy.dim(),proxy.getlen())"<<endl;
            return EuclideanDistanceAVX(proxy[i],proxy[j],proxy.dim(),proxy.getlen());
        #else
            //cout<<"EuclideanDistanceAVX(proxy[i],proxy[j],proxy.dim())"<<endl;
            return EuclideanDistanceAVX(proxy[i],proxy[j],proxy.dim());
        #endif
        #endif
    #else
        //cout<<"float_l2sqr_avx(proxy[i],proxy[j],proxy.dim())"<<endl;
        return float_l2sqr_avx(proxy[i],proxy[j],proxy.dim());
        // return EuclideanDistanceAVX(proxy[i],proxy[j],proxy.dim());
    #endif
    #endif
#endif

    }
    void test()
    {
        // size_t stride2=proxy.step();
        // float a=test2(proxy[0],proxy[0],proxy.dim(),stride2);
    }
};
