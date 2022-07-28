
# GemmEx VS Copy+SGEMM Benchmark

This code benchmarks the case of a GEMM using two inputs in half precision with the output expected to be in single precision. One method uses GemmEx to with A and B in half precision with C being single precision. The second method uses a copy kernel to convert A and B from half to single and then performing a single precision GEMM. 

# Running

The codes can be built using the included Makefile. This has been tested using Cuda 11.1.0 with GCC 8.3.0.

The benchmarks can be run individually with 

```
#build
make clean && make 

#GemmEx
./gemmEx --m 1024 --n 1024 --k 1024 --n-runs 10

#Copy+SGEMM
./gemmEx --m 1024 --n 1024 --k 1024 --n-runs 10 --copy
```

# Results

Reproduction of these results can be achieved using the included `run_bench.sh`.

```
bash run_bench.sh
```

These were gathered on a Power9 System using a single V100 gpu.

| M,N,K | Copy+SGemm   | GemmEx       |
| ----- | ------------ | ------------ |
| 512   | 0.00010936s  | 5.04032e-05s |
| 1024  | 0.00387891s  | 3.19456e-05s |
| 2048  | 0.00545101s  | 6.38176e-05s |
| 4096  | 0.00802283s  | 0.000226691s |
| 8192  | 0.0418808s   | 0.00125228s  |
| 16284 | 0.120269s    | 0.00859056s  | 
| 32768 | 0.606295s    | 0.105581s    |