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
- Sort all the numbers using bitonic sort.
- Each warp counts up different number by counting continuous length of it.
- Applying max-reduction on the count of each number.
