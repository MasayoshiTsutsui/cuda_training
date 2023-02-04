# CUDA Util Codes

## Mode calculation
It calculates the mode of random numbers all on GPU and measure its exec time.

### How to Compile
```
nvcc mode.cu
```

### Restrictions
All process must be performed on GPU.
However, only the random numbers' initialization is performed on CPU.

### Calculation Flow
- Phase1: Sort all the numbers using bitonic sort.
- Phase2: Each warp counts up different number by counting continuous length of it.
- Phase3: Applying max-reduction on the count of each number.

### Execution time
comiler: nvcc V11.1.74

Hardware: NVIDIA A100
- Phase1: 7.7922 msec.
- Phase2: 0.439296 msec.
- Phase3: 2.00684 msec.
