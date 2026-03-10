# Makefile for mphil_dis_cholesky
#
# Usage:
#   make                                          build default version (v1_baseline)
#   make VERSION=v2_serial_opt                    build a specific version
#   make test                                     build and run correctness tests
#   make bench                                    build benchmark binary
#   make bench VERSION=v5_serial_blocked NB=64    sweep panel width (serial)
#   make bench VERSION=v5_openmp_blocked NB=128   sweep panel width (parallel)
#   make clean                                    remove all build artefacts
#
# Available versions (src/cholesky_<VERSION>.c):
#   v1_baseline          exact spec loop, -O0 (baseline)
#   v2_serial_opt        loop interchange + c_ip hoist + inv_diag, -O3
#   v3_serial_opt        v2 with explicit reciprocal division, -O3
#   v3_openmp            first OpenMP parallel version (omp for, static schedule)
#   v5_serial_blocked    panel-blocked serial reference (tune panel with NB=N)
#   v5_openmp_blocked    panel-blocked OpenMP (tune panel with NB=N)
#
# To add a new version: create src/cholesky_<name>.c and add an ifeq block below.

VERSION  ?= v1_baseline
NB       ?= 96    # panel width for v5_*_blocked; NB=96 is empirically optimal on CSD3 icelake

# Base flags — always applied regardless of version
BASE_CFLAGS = -Wall -Wextra -std=gnu11 -I include

# Per-version flags (add optimisation and OpenMP as needed)
ifeq ($(VERSION),v1_baseline)
  OPT_FLAGS = -O0 -g
endif
ifeq ($(VERSION),v2_serial_opt)
  OPT_FLAGS = -O3 -march=native -ffast-math
endif
ifeq ($(VERSION),v3_serial_opt)
  OPT_FLAGS = -O3 -march=native -ffast-math
endif
ifeq ($(VERSION),v3_openmp)
  OPT_FLAGS = -O3 -march=native -ffast-math -fopenmp
endif
ifeq ($(VERSION),v5_serial_blocked)
  OPT_FLAGS = -O3 -march=native -ffast-math -DBLOCK_NB=$(NB)
endif
ifeq ($(VERSION),v5_openmp_blocked)
  OPT_FLAGS = -O3 -march=native -ffast-math -fopenmp -DBLOCK_NB=$(NB)
endif

# If VERSION is not listed above, fall back to -O3 (unknown version still compiles)
OPT_FLAGS ?= -O3 -march=native -ffast-math

CFLAGS = $(BASE_CFLAGS) $(OPT_FLAGS)

CC      = gcc
LDFLAGS = -lm

LIB_SRC  = src/cholesky_$(VERSION).c
LIB_OBJ  = src/cholesky_$(VERSION).o
LIB_A    = lib/libcholesky.a

EXAMPLE_BIN = example/example
TEST_BIN    = test/test_correctness
BENCH_BIN   = test/benchmark

.PHONY: all lib example test bench clean

all: lib example $(TEST_BIN) $(BENCH_BIN)

$(LIB_OBJ): $(LIB_SRC) include/mphil_dis_cholesky.h
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB_A): $(LIB_OBJ)
	mkdir -p lib
	ar rcs $@ $<

lib: $(LIB_A)

$(EXAMPLE_BIN): example/example.c $(LIB_A)
	$(CC) $(CFLAGS) $< -Llib -lcholesky $(LDFLAGS) -o $@

example: $(EXAMPLE_BIN)

$(TEST_BIN): test/test_correctness.c $(LIB_A)
	$(CC) $(CFLAGS) $< -Llib -lcholesky $(LDFLAGS) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

$(BENCH_BIN): test/benchmark.c $(LIB_A)
	$(CC) $(CFLAGS) $< -Llib -lcholesky $(LDFLAGS) -o $@

bench: $(BENCH_BIN)

clean:
	rm -f src/*.o $(LIB_A) $(EXAMPLE_BIN) $(TEST_BIN) $(BENCH_BIN)
	rm -rf lib/
