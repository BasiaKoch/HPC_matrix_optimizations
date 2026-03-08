# Makefile for mphil_dis_cholesky
#
# Usage:
#   make                        build default version (v1_baseline)
#   make VERSION=v2_serial_opt  build a specific version
#   make test                   build and run correctness tests
#   make bench                  build benchmark binary
#   make clean                  remove all build artefacts
#
# Available versions (src/cholesky_<VERSION>.c):
#   v1_baseline      exact spec loop, no optimisation
#   v2_serial_opt    loop reorder, precompute 1/diag, -O3
#   v3_openmp        basic OpenMP parallelisation
#   v4_tuned         OpenMP schedule tuning

VERSION  ?= v1_baseline

# Flags per version
ifeq ($(VERSION),v1_baseline)
  CFLAGS = -O0 -g -Wall -Wextra -std=c11 -I include
endif
ifeq ($(VERSION),v2_serial_opt)
  CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -std=c11 -I include
endif
ifeq ($(VERSION),v3_openmp)
  CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall -Wextra -std=c11 -I include
endif
ifeq ($(VERSION),v4_tuned)
  CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall -Wextra -std=c11 -I include
endif

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
