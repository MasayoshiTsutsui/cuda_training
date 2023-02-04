//--コンパイル方法--
// nvcc mode.cu
// 以下のバージョンでコンパイルができることを確認
// nvcc: NVIDIA (R) Cuda compiler driver
// Copyright (c) 2005-2020 NVIDIA Corporation
// Built on Tue_Sep_15_19:10:02_PDT_2020
// Cuda compilation tools, release 11.1, V11.1.74
// Build cuda_11.1.TC455_06.29069683_0


#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
using namespace std;

#define MAXSEED 10 // increase this if you want to take more time data measurings.
#define INTMAX 9999
#define WARPSIZE 32
#define LOG2WARP 5 //log2(32)
#define WARP_BLOCK 4 //warps per block
#define THREAD_BLOCK 128 //WARPSIZE * WARP_BLOCK
#define NUMS_WARP 256 // the number of data checked by a warp
#define NUMS_BLOCK 1024 //NUMS_WARP * WARP_BLOCK
#define MODEMINCNT 16 // N / M (same as minPossibleCnt in main)


void errorCheck(cudaError_t ret);
__device__ int nextUniqueIdx(int localtid, int localwid, uint64_t *nums, int *tmp, int baseidx, int N);
__global__ void countingKernel(uint64_t *nums, int N, uint64_t *uniqueNums, int *numsCnt);
void generate_testcase(uint64_t* a, int seed, int N, int M);
uint64_t calcModeHost(uint64_t *a, int N);


int main() {
    uint64_t hostAns = 0; // mode calculated by CPU
    uint64_t deviceAns = 0; // calculated by GPU
    const int N = 1 << 24;
    const int M = 1 << 20;
    const int minPossibleCnt = N / M; // the mode's counts are at least 16 (N / M).
    uint64_t* nums_h = (uint64_t*)malloc(sizeof(uint64_t) * N);
    uint64_t* nums_d;
    uint64_t* uniqueNums_d; // keeps unique numbers which appears more than 16(minPossibleCnt) times.
    int* numsCnt_d; // keeps the counts of uniqueNums_d.

    float measuredTimes[3];
    float averagedTimes[3] = {0, 0, 0};

    errorCheck(cudaMalloc((void**)&nums_d, sizeof(uint64_t) * N));
    errorCheck(cudaMalloc((void**)&uniqueNums_d, sizeof(uint64_t) * N / minPossibleCnt)); // since only counts nums appearing 16(minPossibleCnt) times, N / minPossibleCnt is enough.
    errorCheck(cudaMalloc((void**)&numsCnt_d, sizeof(int) * N / minPossibleCnt));

    thrust::device_ptr<uint64_t> numsDevptr = thrust::device_pointer_cast(nums_d); //ptr for thrust
    thrust::device_ptr<int> numsCntDevptr = thrust::device_pointer_cast(numsCnt_d); //ptr for thrust
    thrust::device_ptr<uint64_t> uniqueNumsDevptr = thrust::device_pointer_cast(uniqueNums_d); //ptr for thrust

    // time recorders
    cudaEvent_t start1, stop1, start2, stop2, start3, stop3;
    errorCheck(cudaEventCreate(&start1));
    errorCheck(cudaEventCreate(&stop1));
    errorCheck(cudaEventCreate(&start2));
    errorCheck(cudaEventCreate(&stop2));
    errorCheck(cudaEventCreate(&start3));
    errorCheck(cudaEventCreate(&stop3));

    for (int seed = 0; seed <= MAXSEED; seed++) {
        // generate inputs
        generate_testcase(nums_h, seed, N, M);

        // calculate mode on CPU
        hostAns = calcModeHost(nums_h, N);

        // transfer inputs to device
        errorCheck(cudaMemcpy(nums_d, nums_h, sizeof(uint64_t) * N, cudaMemcpyHostToDevice));

        //phase1 : sorting
        cudaEventRecord(start1);

        thrust::sort(numsDevptr, numsDevptr + N);

        errorCheck(cudaEventRecord(stop1));
        errorCheck(cudaEventSynchronize(stop1));
        errorCheck(cudaEventElapsedTime(&measuredTimes[0], start1, stop1));

        // phase2 : counting unique numbers
        errorCheck(cudaEventRecord(start2));
        countingKernel<<< N / NUMS_BLOCK, THREAD_BLOCK >>>(nums_d, N, uniqueNums_d, numsCnt_d);

        errorCheck(cudaEventRecord(stop2));
        errorCheck(cudaEventSynchronize(stop2));
        errorCheck(cudaEventElapsedTime(&measuredTimes[1], start2, stop2));

        // phase3 : find mode by max-reduction of unique numbers' count. (more strictly, find offset where the mode is in uniqueNums_d)
        errorCheck(cudaEventRecord(start3));
        thrust::device_ptr<int> modeCntDevptr = thrust::max_element(numsCntDevptr, numsCntDevptr + N / minPossibleCnt);
        // thrust::max_element returns the "first" iterator i which points the largest value (from thrust documentation)
        // so this will return the smallest mode.
        errorCheck(cudaEventRecord(stop3));
        errorCheck(cudaEventSynchronize(stop3));
        errorCheck(cudaEventElapsedTime(&measuredTimes[2], start3, stop3));

        int modeIdx = modeCntDevptr - numsCntDevptr; // offset of the mode's position in uniqueNums_d (and numsCnt_d)

        errorCheck(cudaMemcpy(&deviceAns, uniqueNums_d + modeIdx, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        
        if (hostAns != deviceAns) { // verify the calculated mode
            cout << "host answer and device answer doesn't match in seed " << seed << "!" << endl;
            cout << "host answer :" << hostAns << endl;
            cout << "device answer :" << deviceAns << endl;
        }

        if (seed != 0) { // discard fist data considering the warming up of GPU.
            averagedTimes[0] += measuredTimes[0];
            averagedTimes[1] += measuredTimes[1];
            averagedTimes[2] += measuredTimes[2];
        }
    }

    averagedTimes[0] /= MAXSEED;
    averagedTimes[1] /= MAXSEED;
    averagedTimes[2] /= MAXSEED;

    cout << "// average execution time of " << MAXSEED << " seeds //" << endl;
    cout << "phase1 (sorting) : " << averagedTimes[0] << " msec." << endl;
    cout << "phase2 (counting) : " << averagedTimes[1] << " msec." << endl;
    cout << "phase3 (reduction) : " << averagedTimes[2] << " msec." << endl;
    cout << "total : " << averagedTimes[0] + averagedTimes[1] + averagedTimes[2] << " msec." << endl;


    errorCheck(cudaEventDestroy(start1));
    errorCheck(cudaEventDestroy(stop1));
    errorCheck(cudaEventDestroy(start2));
    errorCheck(cudaEventDestroy(stop2));
    errorCheck(cudaEventDestroy(start3));
    errorCheck(cudaEventDestroy(stop3));


    return 0;
}

void errorCheck(cudaError_t ret) {
    if (ret != cudaSuccess) {
        printf("CUDA Error:%s\n", cudaGetErrorString(ret));
        exit(-1);
    }
}

// calculates the next unique number's idx in a sequential(sorted) numbers.
__device__ int nextUniqueIdx(int localtid, int localwid, uint64_t *nums, int *tmp, int baseidx, int N) {
    if (baseidx + localtid + 1 >= N) { // boundary check
        tmp[threadIdx.x] = localtid + 1;
    }
    else if (nums[baseidx] == nums[baseidx + localtid + 1]) { 
        tmp[threadIdx.x] = INTMAX;
    }
    else {
        tmp[threadIdx.x] = localtid + 1;
    }

    int activeThreads = WARPSIZE / 2;

    for (int i = 0; i < LOG2WARP; i++) { // find min in tmp, which is the localtid of next unique number holder.
        if (localtid < activeThreads) {
            if (tmp[threadIdx.x] > tmp[threadIdx.x+activeThreads]) {
                tmp[threadIdx.x] = tmp[threadIdx.x+activeThreads];
            }
        }
        activeThreads /= 2;
    }
    
    return tmp[localwid * WARPSIZE];
}

__global__ void countingKernel(uint64_t *nums, int N, uint64_t *uniqueNums, int *numsCnt){
    __shared__ int tmp[THREAD_BLOCK]; //working table

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int wid = tid / WARPSIZE;
    int localtid = threadIdx.x % WARPSIZE;
    int localwid = threadIdx.x / WARPSIZE;
    int originBaseidx = wid * NUMS_WARP;
    int baseidx = wid * NUMS_WARP;

    int cntIdx = wid * (NUMS_WARP / MODEMINCNT); //current idx of uniqueNums & numsCnt

    if (localtid / (NUMS_WARP / MODEMINCNT) == 0) {
        uniqueNums[cntIdx + localtid] = 0;
        numsCnt[cntIdx + localtid] = 0;
    }

    if (baseidx != 0 && nums[baseidx] == nums[baseidx-1]) { // if the previous num == start num, then  the previous warp is reponsible for this area
        while(1) { // move baseidx until next unique number appears.
            if (baseidx >= N) {
                return;
            }
            int nextUniqIdx = nextUniqueIdx(localtid, localwid, nums, tmp, baseidx, N);
            if (nextUniqIdx == INTMAX) { // when all 32(warpsize) numbers are same
                baseidx += WARPSIZE;
            }
            else { 
                baseidx += nextUniqIdx;
                break;
            }
        }
    }

    bool carryFlag = false; // when the unique number has more than 32, the flag rise and keep on counting.

    while (baseidx < originBaseidx + NUMS_WARP || carryFlag) {
        int nextUniqIdx = nextUniqueIdx(localtid, localwid, nums, tmp, baseidx, N);
        if (nextUniqIdx == INTMAX) { // when all 32(warpsize) numbers are same
            if (localtid == 0) {
                uniqueNums[cntIdx] = nums[baseidx];
                numsCnt[cntIdx] += WARPSIZE;
            }
            baseidx += WARPSIZE;
            carryFlag = true;
        }
        else { 
            if (!carryFlag && nextUniqIdx < MODEMINCNT) { // when the number count is under 16, it cannot be the mode.
                baseidx += nextUniqIdx;
                carryFlag = false;
            }
            else {
                if (localtid == 0) {
                    uniqueNums[cntIdx] = nums[baseidx];
                    numsCnt[cntIdx] += nextUniqIdx;
                }
                cntIdx++;
                baseidx += nextUniqIdx;
                carryFlag = false;
            }
        }
    }
}

void generate_testcase(uint64_t* a, int seed, int N, int M) {
    mt19937 mt(seed);
    uniform_int_distribution<uint64_t> dist1;
    vector<uint64_t> x(M);
    for (int i = 0; i < M; i++) {
        x[i] = dist1(mt);
    }
    uniform_int_distribution<int> dist2(0, M-1);
    for (size_t i = 0; i < N; i++) {
        a[i] = x[dist2(mt)];
    }
}

uint64_t calcModeHost(uint64_t *a, int N) {
    unordered_map<uint64_t, int> numcnt; 
    for (int i = 0; i < N; i++) {
        if (numcnt.find(a[i]) == numcnt.end()) {
            numcnt.emplace(a[i], 1);
        }
        else {
            numcnt.at(a[i])++;
        }
    }
    uint64_t mode = 0;
    int modeCnt = 0;
    for (auto itr = numcnt.begin(); itr != numcnt.end(); itr++) {
        if (modeCnt == itr->second && mode > itr->first) {
            mode = itr->first;
            modeCnt = itr->second;
        }
        else if (modeCnt < itr->second) {
            mode = itr->first;
            modeCnt = itr->second;
        }
    }
    return mode;
}