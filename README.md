# Hardware-Conscious Performance Engineering

[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![Hardware Architecture](https://img.shields.io/badge/Architecture-x86__64%20%7C%20AVX2%20%7C%20FMA-orange.svg)]()
[![Optimization](https://img.shields.io/badge/Optimization-SIMD%20%7C%20Cache%20Tiling%20%7C%20Prefetching-green.svg)]()
[![IIT Bombay](https://img.shields.io/badge/IIT%20Bombay-CS683%20Adv%20Comp%20Arch-red.svg)](https://www.cse.iitb.ac.in/)

> **Academic Affiliation**: Course Project for **CS 683: Advanced Computer Architecture**, IIT Bombay  
> **Collaborators**: **Dheeraj Kumar Maradana** **Sabil Ahmad**  **Abhineet Majety**  **B Shashank** 

---

## 📌 Executive Summary

This repository implements rigorous, hardware-conscious kernel engineering for compute-intensive workloads. Starting from correct, naive C++ baselines, we systematically optimize algorithms to extract near-peak utilization from the modern x86 CPU memory hierarchy, instruction-level parallelism (ILP), and SIMD execution units (AVX2/FMA).

The project tackles two critical high-performance compute kernels:
1. **2D Convolution Engine**: Progressive optimization via loop interchange, register unrolling, L1D cache tiling, and handwritten 256-bit AVX2 SIMD intrinsics with Fused Multiply-Add (`_mm256_fmadd_ps`).
2. **SGEMM Matrix Multiplication in `llama.cpp`**: Custom high-throughput single-precision general matrix multiplication (SGEMM) kernel featuring cache-blocking, register-tiled micro-kernels, and software prefetching, directly integrated into the `llama.cpp` inference engine.

---


## 🔬 Key Engineering Implementations

### Task 1: 2D Convolution Optimization (`task1/`)
* **Loop Interchange (`conv_reorder`)**: Transformed 4D iteration space from $(\text{oy}, \text{ox}, \text{ky}, \text{kx})$ to $(\text{oy}, \text{ky}, \text{kx}, \text{ox})$, enforcing unit-stride streaming memory accesses on both input and output buffers, maximizing 64-byte cache line utilization.
* **Loop Unrolling (`conv_unroll`)**: Unrolled innermost compute loops across multiple registers to saturate multiple execution ports and eliminate branch overhead.
* **Cache Tiling (`conv_tile`)**: Blocked input tensors to fit precisely within the 48 KB L1 Data Cache, driving down L1-D misses per kilo-instructions (MPKI).
* **SIMD Intrinsics (`conv_simd`)**: Implemented handwritten AVX2 vector routines utilizing 256-bit `__m256` registers, processing 8 floating-point operations per instruction cycle.

### Task 2: High-Performance SGEMM in `llama.cpp` (`task2/`)
* **Micro-kernel Tiling**: Partitioned matrices into $M_R \times N_R$ register blocks and $M_C \times K_C \times N_C$ cache slices matching CPU cache capacities.
* **Instruction Scheduling & Accumulation**: Maintained 16 SIMD registers simultaneously as accumulation banks to minimize register spills.
* **Software Prefetching**: Inserted targeted `_mm_prefetch` intrinsics for upcoming cache lines to overlap memory transfer latency with floating-point computations.
* **End-to-End LLM Inference**: Integrated the tuned SGEMM kernel into `llama.cpp` for accelerated quantized and FP32 prompt processing and token generation.

---

## 📊 Experimental Setup & Profiling

All hardware counters were evaluated using isolated Linux `perf` harnesses pinned to physical performance cores:
* **CPU**: 13th Gen Intel Core i5-13500H (Raptor Lake, Performance Core pinned via `taskset -c 0`)
* **Caches**: L1D: 48 KB, L2: 2 MB, L3: 18 MB shared
* **Metrics Tracked**: Instructions, Cycles, IPC (Instructions Per Cycle), L1-dcache load misses, L1D MPKI.

---

## 🚀 Building & Running

### Prerequisites
* Linux (x86_64) with AVX2 and FMA support
* GCC / Clang (`g++ -std=c++17 -mavx2 -mfma -O2`)
* Python 3 + Matplotlib (for automated benchmarking and plotting)

### Task 1: 2D Convolution
```bash
cd task1
make -j$(nproc)

# Run benchmark across all implementations
./bin/conv naive
./bin/conv reorder
./bin/conv unroll
./bin/conv tile
./bin/conv simd

# Run comprehensive automated hardware profiling
python3 benchmark_1a.py
python3 benchmark_1b.py
```

### Task 2: SGEMM & llama.cpp
```bash
cd task2
make -j$(nproc)
./bin/benchmark_sgemm
```

---

## 👥 Contributors & Academic Context

Developed as part of **CS 683 (Advanced Computer Architecture)** at the **Indian Institute of Technology Bombay**.
* **Dheeraj Kumar Maradana** ([@dheerajkumar2005](https://github.com/dheerajkumar2005))
* **Sabil Ahmad** ([@sabilxD](https://github.com/sabilxD))
* **Abhineet Majety** ([@abhineetm13](https://github.com/abhineetm13))
* **B Shashank** ([@Shasankbt](https://github.com/Shasankbt))
