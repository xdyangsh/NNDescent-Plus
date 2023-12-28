#include <vector>
#include <time.h>
#include <iostream>
#include <cstring>
#include <malloc.h>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#define BIT_NUM 14
#ifdef __AVX2__
#define KGRAPH_MATRIX_ALIGN 32
#else
#ifdef __SSE__
#define KGRAPH_MATRIX_ALIGN 16
#else
#define KGRAPH_MATRIX_ALIGN 16
#endif
#endif

#ifdef __AVX2__
#ifdef GROUP
#define group_num 8
#else
#ifdef ALL
#define group_num 1
#else
#define group_num 8
#endif
#endif
#else
#ifdef __SSE__
#ifdef GROUP
#define group_num 4
#else
#ifdef ALL
#define group_num 1
#else
#define group_num 4
#endif
#endif
#else
#ifdef GROUP
#define group_num 8
#else
#ifdef ALL
#define group_num 1
#else
#define group_num 8
#endif
#endif
#endif
#endif

 int *decoding(const unsigned char *data,unsigned dim,size_t stride)
{
    unsigned stride2 = (sizeof(int) * dim + KGRAPH_MATRIX_ALIGN - 1) / KGRAPH_MATRIX_ALIGN * KGRAPH_MATRIX_ALIGN;
    int *tempdata=nullptr;
    tempdata = (int *)memalign(KGRAPH_MATRIX_ALIGN, stride2); // SSE instruction needs data to be aligned
    if (!tempdata) throw std::runtime_error("memalign");
    memset(tempdata, 0, stride2);
    int currentLen=8;
    int current=stride-1;
    unsigned char temp=data[current];
    for(int i=0;i<dim;i++)
    {
        int current_bit=0;
        while(current_bit<BIT_NUM)
        {
            if(currentLen>0&&(BIT_NUM-current_bit)>=currentLen)
            {
                unsigned short mask = (1 << currentLen) - 1;
                tempdata[i]=(tempdata[i]) | ((temp & mask) << current_bit);
                temp=temp>>currentLen;
                current_bit+=currentLen;
                currentLen=0;
            }
            else if(currentLen>0&&(BIT_NUM-current_bit)<currentLen)
            {
                unsigned short mask = (1 << (BIT_NUM-current_bit)) - 1;
                tempdata[i]=(tempdata[i]) | ((temp & mask) << current_bit);
                temp=temp>>(BIT_NUM-current_bit);
                currentLen-=(BIT_NUM-current_bit);
                current_bit=BIT_NUM;
            }
            else
            {
                current--;
                temp=data[current];
                currentLen=8;
            }
        }
    }
    // for(int i=0;i<dim;i++)
    // {
    //     cout<<tempdata[i]<<" ";
    // }
    return tempdata;
}

float test2(unsigned char *lhs, unsigned char *rhs,unsigned dim,size_t stride)
{
    decoding(lhs,dim,stride);
    return 0;
}

float EuclideanDistance(float const *lhs, float const *rhs,unsigned dim){
        float ans = 0.0;
        std::cout<<"dim:"<<dim<<std::endl;
        for (unsigned i = 0; i < dim; ++i) {
            float v = float(lhs[i]) - float(rhs[i]);
            v *= v;
            ans += v;
        }
        return ans;
}

unsigned int EuclideanDistance(unsigned short const *lhs, unsigned short const *rhs,unsigned dim){
        unsigned int ans = 0;
        //std:://cout<<"dim:"<<dim<<std::endl;
        for (unsigned i = 0; i < dim; ++i) {
            unsigned int v = lhs[i] - rhs[i];
            v *= v;
            ans += (unsigned int)v;
        }
        return ans;
}

unsigned int EuclideanDistance(unsigned char const *lhs, unsigned char const *rhs,unsigned dim){
        unsigned int ans = 0;
        //std:://cout<<"this one"<<dim<<std::endl;
        for (unsigned i = 0; i < dim; ++i) {
            unsigned int v = (unsigned short)lhs[i] - (unsigned short)rhs[i];
            v *= v;
            ans += (unsigned int)v;
        }
        return ans;
}


float EuclideanDistance(unsigned short const *lhs, unsigned short const *rhs,unsigned dim,float const* length){
        float sum=0;
        //std:://cout<<"dim:"<<dim<<std::endl;
        int a=0,b=0;
        a=dim%group_num;
        b=dim/group_num;
        for (unsigned i = 0; i < b; ++i) {
            unsigned int ans = 0;
            for(unsigned j=0; j < group_num; ++j)
            {
                unsigned int v = (lhs[i*group_num+j] - rhs[i*group_num+j]);
                v *= v;
                ans += v;
            }
            sum+=ans*length[i*group_num]*length[i*group_num];
        }
        unsigned int ans = 0;
        for(unsigned j=0;j<a;++j)
        {
            unsigned int v = (lhs[b*group_num+j] - rhs[b*group_num+j]);
            v *= v;
            ans += v;
        }
        sum+=ans*length[b*group_num]*length[b*group_num];
        return sum;
}

float EuclideanDistance(unsigned char const *lhs, unsigned char const *rhs,unsigned dim,float const* length){
        float sum=0;
        //std:://cout<<"dim:"<<dim<<std::endl;
        int a=0,b=0;
        a=dim%group_num;
        b=dim/group_num;
        for (unsigned i = 0; i < b; ++i) {
            unsigned int ans = 0;
            for(unsigned j=0; j < group_num; ++j)
            {
                unsigned int v = (unsigned short)lhs[i*group_num+j] - (unsigned short)rhs[i*group_num+j];
                v *= v;
                ans += v;
            }
            sum+=ans*length[i*group_num];
        }
        unsigned int ans = 0;
        for(unsigned j=0;j<a;++j)
        {
            unsigned int v = (unsigned short)lhs[b*group_num+j] - (unsigned short)rhs[b*group_num+j];
            v *= v;
            ans += v;
        }
        sum+=ans*length[b*group_num];
        return sum;
}

#ifdef __AVX512F__
#define AVX512_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm512_loadu_ps(addr1);\
    tmp2 = _mm512_loadu_ps(addr2);\
    tmp1 = _mm512_sub_ps(tmp1, tmp2); \
    tmp1 = _mm512_mul_ps(tmp1, tmp1); \
    dest = _mm512_add_ps(dest, tmp1); 
float float_l2sqr_avx512 (float const *t1, float const *t2, unsigned dim) {
    //cout<<"float_l2sqr_avx512"<<endl;
    __m512 sum;
    __m512 l0, l1, l2, l3;
    __m512 r0, r1, r2, r3;
    unsigned D = (dim + 15) & ~15U; // # dim aligned up to 256 bits, or 8 floats
    unsigned DR = D % 64;
    unsigned DD = D - DR;
    const float *l = t1;
    const float *r = t2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[16] __attribute__ ((aligned (64))) = {0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm512_load_ps(unpack);
    switch (DR) {
        case 48:
            AVX512_L2SQR(e_l+32, e_r+32, sum, l2, r2);
        case 32:
            AVX512_L2SQR(e_l+16, e_r+16, sum, l1, r1);
        case 16:
            AVX512_L2SQR(e_l, e_r, sum, l0, r0);
    }
    for (unsigned i = 0; i < DD; i += 64, l += 64, r += 64) {
        AVX512_L2SQR(l, r, sum, l0, r0);
        AVX512_L2SQR(l + 16, r + 16, sum, l1, r1);
        AVX512_L2SQR(l + 32, r + 32, sum, l2, r2);
        AVX512_L2SQR(l + 48, r + 48, sum, l3, r3);
    }
    _mm512_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7]
        + unpack[8] + unpack[9] + unpack[10] + unpack[11]
        + unpack[12] + unpack[13] + unpack[14] + unpack[15];
    return ret;//sqrt(ret);
}
#endif

#ifdef __AVX2__
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_loadu_ps(addr1);\
    tmp2 = _mm256_loadu_ps(addr2);\
    tmp1 = _mm256_sub_ps(tmp1, tmp2); \
    tmp1 = _mm256_mul_ps(tmp1, tmp1); \
    dest = _mm256_add_ps(dest, tmp1); 
float float_l2sqr_avx (float const *t1, float const *t2, unsigned dim) {
    //cout<<"float_l2sqr_avx"<<endl;
    __m256 sum;
    __m256 l0, l1, l2, l3;
    __m256 r0, r1, r2, r3;
    unsigned D = (dim + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
    unsigned DR = D % 32;
    unsigned DD = D - DR;
    const float *l = t1;
    const float *r = t2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm256_load_ps(unpack);
    switch (DR) {
        case 24:
            AVX_L2SQR(e_l+16, e_r+16, sum, l2, r2);
        case 16:
            AVX_L2SQR(e_l+8, e_r+8, sum, l1, r1);
        case 8:
            AVX_L2SQR(e_l, e_r, sum, l0, r0);
    }
    for (unsigned i = 0; i < DD; i += 32, l += 32, r += 32) {
        AVX_L2SQR(l, r, sum, l0, r0);
        AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
        AVX_L2SQR(l + 16, r + 16, sum, l2, r2);
        AVX_L2SQR(l + 24, r + 24, sum, l3, r3);
    }
    _mm256_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    return ret;//sqrt(ret);
}
#endif

#ifdef __SSE2__
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);\
    tmp2 = _mm_load_ps(addr2);\
    tmp1 = _mm_sub_ps(tmp1, tmp2); \
    tmp1 = _mm_mul_ps(tmp1, tmp1); \
    dest = _mm_add_ps(dest, tmp1); 
float float_l2sqr_sse2 (float const *t1, float const *t2, unsigned dim) {
    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (dim + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = t1;
    const float *r = t2;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__ ((aligned (16))) = {0, 0, 0, 0};
    float ret = 0.0;
    sum = _mm_load_ps(unpack);
    switch (DR) {
        case 12:
            SSE_L2SQR(e_l+8, e_r+8, sum, l2, r2);
        case 8:
            SSE_L2SQR(e_l+4, e_r+4, sum, l1, r1);
        case 4:
            SSE_L2SQR(e_l, e_r, sum, l0, r0);
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        SSE_L2SQR(l, r, sum, l0, r0);
        SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
        SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
        SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
    return ret;//sqrt(ret);
}
#endif

#ifdef __AVX512F__
float EuclideanDistanceAVX512(const float* lhs, const float* rhs, int dim)
{
    int numIterations = ceil(dim / 16);
    __m512 sum;
    float unpack[16] __attribute__ ((aligned (64))) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm512_load_ps(unpack);
    for (int i = 0; i < numIterations; i++)
    {
        __m512 A = _mm512_loadu_ps(lhs + i * 16);
        __m512 B = _mm512_loadu_ps(rhs + i * 16);
        __m512 difference = _mm512_sub_ps(A, B);
        __m512 squaredDifference = _mm512_mul_ps(difference, difference);
        sum = _mm512_add_ps(sum, squaredDifference);
    }
    _mm512_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7]
        + unpack[8] + unpack[9] + unpack[10] + unpack[11]
        + unpack[12] + unpack[13] + unpack[14] + unpack[15];
    return ret;
}

float EuclideanDistanceAVX512(const unsigned char* lhs, const unsigned char* rhs, unsigned dim)
{
    unsigned numIterations = ceil(dim/16);
    
    __m512 sum;
    float unpack[16] __attribute__ ((aligned (64))) = {0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm512_load_ps(unpack);
    for (int i = 0; i < numIterations; i++)
    {
        __m128i A = _mm_loadu_si128((__m128i*)(lhs + i * 16));
        __m128i B = _mm_loadu_si128((__m128i*)(rhs + i * 16));
        
        __m512 AFloat = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(A));
        __m512 BFloat = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(B));
        
        __m512 difference = _mm512_sub_ps(AFloat, BFloat);
        
        __m512 squaredDifference = _mm512_mul_ps(difference, difference);
        
        sum = _mm512_add_ps(sum, squaredDifference);
    }
    _mm512_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7]
        + unpack[8] + unpack[9] + unpack[10] + unpack[11]
        + unpack[12] + unpack[13] + unpack[14] + unpack[15];
        //cout<<"EuclideanDistanceAVX512"<<endl;
    return ret;
}
float EuclideanDistanceAVX512(const unsigned char* lhs, const unsigned char* rhs, unsigned dim,float const* length)
{
    unsigned numIterations = ceil(dim/16);
    
    __m512 sum;
    float unpack[16] __attribute__ ((aligned (64))) = {0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm512_load_ps(unpack);
    for (int i = 0; i < numIterations; i++)
    {
        __m128i A = _mm_loadu_si128((__m128i*)(lhs + i * 16));
        __m128i B = _mm_loadu_si128((__m128i*)(rhs + i * 16));
        __m512 len=_mm512_load_ps(length+i*16);
        __m512 AFloat = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(A));
        __m512 BFloat = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(B));
        
        __m512 difference = _mm512_sub_ps(AFloat, BFloat);
        
        __m512 squaredDifference = _mm512_mul_ps(difference, difference);
        squaredDifference = _mm512_mul_ps(squaredDifference,len);
        sum = _mm512_add_ps(sum, squaredDifference);
    }
    _mm512_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7]
        + unpack[8] + unpack[9] + unpack[10] + unpack[11]
        + unpack[12] + unpack[13] + unpack[14] + unpack[15];
        //cout<<"EuclideanDistanceAVXLength512"<<endl;
    return ret;
}


float EuclideanDistanceGroupAVX512(const unsigned char* lhs, const unsigned char* rhs, unsigned dim,float const* length)
{
    unsigned numIterations = ceil(dim/16);
    
    __m512 sum;
    float unpack[16] __attribute__ ((aligned (64))) = {0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm512_load_ps(unpack);
    for (int i = 0; i < numIterations; i++)
    {
        __m128i A = _mm_loadu_si128((__m128i*)(lhs + i * 16));
        __m128i B = _mm_loadu_si128((__m128i*)(rhs + i * 16));
        
        __m512 AFloat = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(A));
        __m512 BFloat = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(B));
        
        __m512 difference = _mm512_sub_ps(AFloat, BFloat);
        
        __m512 squaredDifference = _mm512_mul_ps(difference, difference);
        
        sum = _mm512_add_ps(sum, squaredDifference);
    }
    __m512 len=_mm512_load_ps(length);
    sum = _mm512_mul_ps(sum,len);
    _mm512_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7]
        + unpack[8] + unpack[9] + unpack[10] + unpack[11]
        + unpack[12] + unpack[13] + unpack[14] + unpack[15];
        //cout<<"EuclideanDistanceGroupAVX512"<<endl;
    return ret;
}

float EuclideanDistanceShortAVX512(const unsigned short* lhs, const unsigned short* rhs, unsigned dim)
{
    unsigned numIterations = ceil(dim / 16);
    __m512 sum,AFloat,BFloat;
    __m256i A,B;
    float unpack[16] __attribute__ ((aligned (64))) = {0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm512_load_ps(unpack);
    for (int i = 0; i < numIterations; i++)
    {
        A = _mm256_loadu_si256((__m256i*)(lhs + i * 16));
        B = _mm256_loadu_si256((__m256i*)(rhs + i * 16));
        __m512 AFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(A));
        __m512 BFloat = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(B));
        __m512 difference = _mm512_sub_ps(AFloat, BFloat);
        __m512 squaredDifference = _mm512_mul_ps(difference, difference);
        sum = _mm512_add_ps(sum, squaredDifference);
    }
    
    _mm512_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7]
        + unpack[8] + unpack[9] + unpack[10] + unpack[11]
        + unpack[12] + unpack[13] + unpack[14] + unpack[15];
    return ret;
}
#endif

#ifdef __AVX2__
float EuclideanDistanceShortAVX(const unsigned short* lhs, const unsigned short* rhs, unsigned dim)
{
    //cout<<"shortAVX"<<endl;
    __m256 sum;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_load_ps(unpack);
    for (int i = 0; i < dim; i+=8)
    {
        __m128i A = _mm_loadu_si128((__m128i*)(lhs + i));
        __m128i B = _mm_loadu_si128((__m128i*)(rhs + i));
        __m256 AFloat = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(A));
        __m256 BFloat = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(B));
        __m256 difference = _mm256_sub_ps(AFloat, BFloat);
        __m256 squaredDifference = _mm256_mul_ps(difference, difference);
        sum = _mm256_add_ps(sum, squaredDifference);
    }
    
    _mm256_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    // unsigned int result=EuclideanDistance(lhs,rhs,dim);
    // cout<<ret<<"  "<<result<<endl;
    return ret;
}

float EuclideanDistanceAVX(const unsigned char* lhs, const unsigned char* rhs, unsigned dim)
{
    unsigned numIterations = ceil(dim/8);
    
    __m256 sum;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_load_ps(unpack);
    for (int i = 0; i < numIterations; i++)
    {
        __m128i A = _mm_loadl_epi64((__m128i*)(lhs + i * 8));
        __m128i B = _mm_loadl_epi64((__m128i*)(rhs + i * 8));
        
        __m256 AFloat = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(A));
        __m256 BFloat = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(B));
        
        __m256 difference = _mm256_sub_ps(AFloat, BFloat);
        
        __m256 squaredDifference = _mm256_mul_ps(difference, difference);
        
        sum = _mm256_add_ps(sum, squaredDifference);
    }
    _mm256_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
        //cout<<"EuclideanDistanceAVX"<<endl;
    return ret;
}

float EuclideanDistanceAVX(const unsigned char* lhs, const unsigned char* rhs, unsigned dim,float const* length)
{
    unsigned numIterations = ceil(dim/8);
    
    __m256 sum;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_load_ps(unpack);
    for (int i = 0; i < numIterations; i++)
    {
        __m128i A = _mm_loadl_epi64((__m128i*)(lhs + i * 8));
        __m128i B = _mm_loadl_epi64((__m128i*)(rhs + i * 8));
        __m256 len=_mm256_load_ps(length+i*8);
        
        __m256 AFloat = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(A));
        __m256 BFloat = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(B));
        
        __m256 difference = _mm256_sub_ps(AFloat, BFloat);
        
        __m256 squaredDifference = _mm256_mul_ps(difference, difference);
        squaredDifference = _mm256_mul_ps(squaredDifference, len);
        sum = _mm256_add_ps(sum, squaredDifference);
    }
    _mm256_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    // float result=EuclideanDistance(lhs,rhs,dim,length);
    // //cout<<ret<<"  "<<result<<endl;
    //cout<<"EuclideanDistanceAVXLength"<<endl;
    return ret;
}
float EuclideanDistanceGroupAVX(const unsigned char* lhs, const unsigned char* rhs, unsigned dim,float const* length)
{
    unsigned numIterations = ceil(dim/8);
    
    __m256 sum;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_load_ps(unpack);
    for (int i = 0; i < numIterations; i++)
    {
        __m128i A = _mm_loadl_epi64((__m128i*)(lhs + i * 8));
        __m128i B = _mm_loadl_epi64((__m128i*)(rhs + i * 8));
        
        __m256 AFloat = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(A));
        __m256 BFloat = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(B));
        
        __m256 difference = _mm256_sub_ps(AFloat, BFloat);
        
        __m256 squaredDifference = _mm256_mul_ps(difference, difference);       
        sum = _mm256_add_ps(sum, squaredDifference);
    }
    __m256 len=_mm256_load_ps(length);
    sum = _mm256_mul_ps(sum, len);
    _mm256_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    // float result=EuclideanDistance(lhs,rhs,dim,length);
    // //cout<<ret<<"  "<<result<<endl;
    //cout<<"EuclideanDistanceGroupAVX"<<endl;
    return ret;
}

float EuclideanDistanceAVX(const float* lhs, const float* rhs, int dim)
{
    //cout<<"2222"<<endl;
    //int numIterations = ceil(dim / 8);
    __m256 sum;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_load_ps(unpack);
    for (int i = 0; i < dim; i+=8)
    {
        __m256 A = _mm256_loadu_ps(lhs + i);
        __m256 B = _mm256_loadu_ps(rhs + i);
        __m256 difference = _mm256_sub_ps(A, B);
        __m256 squaredDifference = _mm256_mul_ps(difference, difference);
        sum = _mm256_add_ps(sum, squaredDifference);
    }
    _mm256_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    return ret;
}
float EuclideanDistanceBitAVX(const unsigned char* LHS, const unsigned char* RHS, unsigned dim, size_t stride)
{
    //cout<<"shortAVX"<<endl;
    int *lhs=decoding(LHS,dim,stride);
    int *rhs=decoding(RHS,dim,stride);
    __m256 sum;
    float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
    sum = _mm256_load_ps(unpack);
    for (int i = 0; i < dim; i+=8)
    {
        __m256i A = _mm256_loadu_si256((__m256i*)(lhs + i));
        __m256i B = _mm256_loadu_si256((__m256i*)(rhs + i));
        __m256 AFloat = _mm256_cvtepi32_ps(A);
        __m256 BFloat = _mm256_cvtepi32_ps(B);
        __m256 difference = _mm256_sub_ps(AFloat, BFloat);
        __m256 squaredDifference = _mm256_mul_ps(difference, difference);
        sum = _mm256_add_ps(sum, squaredDifference);
    }
    
    _mm256_storeu_ps(unpack, sum);
    float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
        + unpack[4] + unpack[5] + unpack[6] + unpack[7];
    // unsigned int result=EuclideanDistance(lhs,rhs,dim);
    // cout<<ret<<"  "<<result<<endl;
    //cout<<ret<<endl;
    free(lhs);
    free(rhs);
    return ret;
}
#endif







// float EuclideanDistanceAVX(const unsigned short* lhs, const unsigned short* rhs, unsigned dim)
// {
//     unsigned numIterations = ceil(dim / 8);
//     __m256 sum,AFloat,BFloat;
//     __m128i A,B;
//     float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
//     sum = _mm256_load_ps(unpack);
//     float result=0;
//     for (int i = 0; i < numIterations; i++)
//     {
//         A = _mm_loadu_si128((__m128i*)(lhs + i * 8));
//         B = _mm_loadu_si128((__m128i*)(rhs + i * 8));
//         __m256 AFloat = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(A));
//         __m256 BFloat = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(B));
//         __m256 difference = _mm256_sub_ps(AFloat, BFloat);
//         sum = _mm256_mul_ps(difference, difference);
//         _mm256_storeu_ps(unpack, sum);
//         float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
//             + unpack[4] + unpack[5] + unpack[6] + unpack[7];
//         result += ret;
//     }
//     return result;
// }


// float EuclideanDistanceAVX(const unsigned short* lhs, const unsigned short* rhs, unsigned dim)
// {
//     unsigned numIterations = ceil(dim / 16);
//     __m256 sum,AFloat,BFloat;
//     __m128i high,low;
//     float unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};
//      sum = _mm256_load_ps(unpack);
//     for (int i = 0; i < numIterations; i++)
//     {
//         __m256i left= _mm256_load_si256((__m256i*)(lhs + i * 16));
//         __m256i right= _mm256_load_si256((__m256i*)(rhs + i * 16));
//         __m256i dest = _mm256_sub_epi16(left, right);
//         high=_mm256_extracti128_si256(dest, 1);
//         low=_mm256_extracti128_si256(dest, 0);
//         __m256 highFloat = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(high));
//         __m256 lowFloat = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(low));
//         __m256 highDifference = _mm256_mul_ps(highFloat, highFloat);
//         __m256 lowDifference = _mm256_mul_ps(lowFloat, lowFloat);
//         sum = _mm256_add_ps(sum, lowDifference);
//         sum = _mm256_add_ps(sum, highDifference);
//     }
    
//     _mm256_storeu_ps(unpack, sum);
//     float ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
//         + unpack[4] + unpack[5] + unpack[6] + unpack[7];
//      unsigned int ans = 0;
//         std:://cout<<"dim:"<<dim<<std::endl;
//         for (unsigned i = 0; i < dim; ++i) {
//             unsigned int v = lhs[i] - rhs[i];
//             v *= v;
//             ans += (unsigned int)v;
//         }
//     //cout<<ret<<"  "<<ans<<endl;
//     return ret;
    
// }
