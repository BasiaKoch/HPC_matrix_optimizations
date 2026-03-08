# Makefile for mphil_dis_cholesky
#
# Usage:
#   make            build library, example, and test binary
#   make lib        build static library only
#   make example    build example program
#   make test       build and RUN correctness tests
#   make clean      remove all build artefacts
#
# Compiler flags:
#   -O0 -g          no optimisation, debug symbols (v0.1-baseline)
#   Change CFLAGS to -O3 -march=native for later optimisation stages.
#
# OpenMP:
#   OpenMP is not used in v0.1-baseline.  To enable it in later stages,
#   add -fopenmp to CFLAGS and LDFLAGS.

CC      = gcc
CFLAGS  = -O0 -g -Wall -Wextra -std=c11 -I include
LDFLAGS = -lm

# ── Directories ────────────────────────────────────────────────────────────
SRC_DIR = src

# ── Library ────────────────────────────────────────────────────────────────
LIB_SRC = $(SRC_DIR)/mphil_dis_cholesky.c
LIB_OBJ = $(SRC_DIR)/mphil_dis_cholesky.o
LIB_A   = lib/libcholesky.a

# ── Example program ────────────────────────────────────────────────────────
EXAMPLE_SRC = example/example.c
EXAMPLE_BIN = example/example

# ── Test program ───────────────────────────────────────────────────────────
TEST_SRC  = test/test_correctness.c
TEST_BIN  = test/test_correctness

BENCH_SRC = test/benchmark.c
BENCH_BIN = test/benchmark

# ── Targets ────────────────────────────────────────────────────────────────
.PHONY: all lib example test bench clean

all: lib example $(TEST_BIN) $(BENCH_BIN)

# Compile library object
$(LIB_OBJ): $(LIB_SRC) include/mphil_dis_cholesky.h
	$(CC) $(CFLAGS) -c $< -o $@

# Archive into static library (create lib/ dir if needed)
$(LIB_A): $(LIB_OBJ)
	mkdir -p lib
	ar rcs $@ $<

lib: $(LIB_A)

# Example program
$(EXAMPLE_BIN): $(EXAMPLE_SRC) $(LIB_A)
	$(CC) $(CFLAGS) $< -Llib -lcholesky $(LDFLAGS) -o $@

example: $(EXAMPLE_BIN)

# Test binary
$(TEST_BIN): $(TEST_SRC) $(LIB_A)
	$(CC) $(CFLAGS) $< -Llib -lcholesky $(LDFLAGS) -o $@

# Build and run correctness tests
test: $(TEST_BIN)
	./$(TEST_BIN)

# Benchmark binary
$(BENCH_BIN): $(BENCH_SRC) $(LIB_A)
	$(CC) $(CFLAGS) $< -Llib -lcholesky $(LDFLAGS) -o $@

bench: $(BENCH_BIN)

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -f $(LIB_OBJ) $(LIB_A) $(EXAMPLE_BIN) $(TEST_BIN) $(BENCH_BIN)
	rm -rf lib/
