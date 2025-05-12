# Simplified Makefile for 2mm OpenMP benchmark
CC = gcc
CFLAGS = -O3 -fopenmp -Wall -march=native
LDFLAGS = -fopenmp

# Build targets
all: 2mm_openmp

2mm_openmp: polybench_2mm_openmp.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f 2mm_openmp.exe

run_small:
	./2mm_openmp 4 0

run_medium:
	./2mm_openmp 4 1

run_large:
	./2mm_openmp 4 2

# Run benchmarks with different thread counts
benchmark:
	@echo Running benchmarks with different thread counts...
	@echo Small dataset (1 thread):
	./2mm_openmp 1 0
	@echo.
	@echo Small dataset (2 threads):
	./2mm_openmp 2 0
	@echo.
	@echo Small dataset (4 threads):
	./2mm_openmp 4 0
	@echo.
	@echo Small dataset (8 threads):
	./2mm_openmp 8 0
	@echo.
	@echo Medium dataset (4 threads):
	./2mm_openmp 4 1
	@echo.
	@echo Large dataset (4 threads):
	./2mm_openmp 4 2