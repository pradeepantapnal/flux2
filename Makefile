# FLUX.2 klein 4B - Pure C Inference Engine

CC = gcc
CUDA_HOME ?= /usr/local/cuda-13.0
NVCC = $(CUDA_HOME)/bin/nvcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math -Iinclude/flux
LDFLAGS = -lm
CUDA_LIB_DIR = $(CUDA_HOME)/targets/x86_64-linux/lib

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CORE_SRCS = \
	src/core/flux.c \
	src/core/flux_kernels.c \
	src/core/flux_tokenizer.c \
	src/core/flux_vae.c \
	src/core/flux_transformer.c \
	src/core/flux_sample.c \
	src/core/flux_image.c \
	src/core/flux_safetensors.c \
	src/core/flux_qwen3.c \
	src/core/flux_qwen3_tokenizer.c \
	src/core/terminals.c
CLI_SRCS = src/cli/flux_cli.c src/core/linenoise.c src/core/embcache.c
APP_SRCS = src/app/main.c
CUDA_SRCS = src/backends/cuda/flux_cuda.cu
SRCS = $(CORE_SRCS) $(CLI_SRCS) $(APP_SRCS)
OBJS = $(SRCS:.c=.o)
CUDA_C_OBJS = $(CORE_SRCS:.c=.cuda.o) $(CLI_SRCS:.c=.cuda.o) src/app/main.cuda.o
CUDA_OBJS = $(CUDA_SRCS:.cu=.cuda.o)
TARGET = flux
LIB = libflux.a

DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address -Iinclude/flux
SANITIZER_COMMON = -Wall -Wextra -O1 -g -fno-omit-frame-pointer -Iinclude/flux -DGENERIC_BUILD

.PHONY: all clean debug lib install info test pngtest help generic blas mps golden-test asan ubsan cuda

all: help

help:
	@echo "FLUX.2 klein 4B - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make generic  - Pure C, no dependencies (slow)"
	@echo "  make blas     - With BLAS acceleration (~30x faster)"
	@echo "  make asan     - Build with Address+Undefined sanitizers (clang)"
	@echo "  make ubsan    - Build with Undefined sanitizer (clang)"
	@echo "  make cuda     - CUDA backend scaffold (CUDA 13.0)"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  make mps      - Apple Silicon with Metal GPU (fastest)"
endif
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Run inference test"
	@echo "  make pngtest  - Compare PNG load on compressed image"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo ""
	@echo "Example: make mps && ./flux -d flux-klein-model -p \"a cat\" -o cat.png"

generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
generic: clean $(TARGET)
	@echo ""
	@echo "Built with GENERIC backend (pure C, no BLAS)"
	@echo "This will be slow but has zero dependencies."

ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend (~30x faster than generic)"

ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
MPS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_METAL -DACCELERATE_NEW_LAPACK
MPS_OBJCFLAGS = $(MPS_CFLAGS) -fobjc-arc
MPS_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation

mps: clean mps-build
	@echo ""
	@echo "Built with MPS backend (Metal GPU acceleration)"

mps-build: $(CORE_SRCS:.c=.mps.o) $(CLI_SRCS:.c=.mps.o) src/app/main.mps.o flux_metal.o
	$(CC) $(MPS_CFLAGS) -o $(TARGET) $^ $(MPS_LDFLAGS)

%.mps.o: %.c
	$(CC) $(MPS_CFLAGS) -c -o $@ $<

flux_shaders_source.h: flux_shaders.metal
	xxd -i $< > $@

flux_metal.o: flux_metal.m include/flux/flux_metal.h flux_shaders_source.h
	$(CC) $(MPS_OBJCFLAGS) -c -o $@ $<

else
mps:
	@echo "Error: MPS backend requires Apple Silicon (arm64)"
	@exit 1
endif
else
mps:
	@echo "Error: MPS backend requires macOS"
	@exit 1
endif

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)
$(LIB): $(CORE_SRCS:.c=.o)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)


asan: CC = clang
asan: CFLAGS = $(SANITIZER_COMMON) -fsanitize=address,undefined
asan: LDFLAGS = -lm -fsanitize=address,undefined
asan: clean $(TARGET)
	@echo ""
	@echo "Built with ASan+UBSan (clang, generic backend)"

ubsan: CC = clang
ubsan: CFLAGS = $(SANITIZER_COMMON) -fsanitize=undefined
ubsan: LDFLAGS = -lm -fsanitize=undefined
ubsan: clean $(TARGET)
	@echo ""
	@echo "Built with UBSan (clang, generic backend)"

CUDA_CFLAGS = $(CFLAGS_BASE) -DUSE_CUDA -I$(CUDA_HOME)/include -Isrc/backends/cuda -DCUDA_HOME_STR=\"$(CUDA_HOME)\"
CUDA_NVFLAGS = -O3 -DUSE_CUDA -Iinclude/flux -I$(CUDA_HOME)/include -Isrc/backends/cuda -Xcompiler "-Wall -Wextra"
CUDA_LDFLAGS = -L$(CUDA_LIB_DIR) -lcudart -lcublas -lcublasLt -lm

cuda: clean cuda-build
	@echo ""
	@echo "Built with CUDA backend scaffold (USE_CUDA)"

cuda-build: $(CUDA_C_OBJS) $(CUDA_OBJS)
	$(NVCC) -o $(TARGET) $^ $(CUDA_LDFLAGS)

%.cuda.o: %.c
	$(CC) $(CUDA_CFLAGS) -c -o $@ $<

%.cuda.o: %.cu
	$(NVCC) $(CUDA_NVFLAGS) -c -o $@ $<

test:
	@python3 run_test.py --flux-binary ./$(TARGET)

test-quick:
	@python3 run_test.py --flux-binary ./$(TARGET) --quick

golden-test:
	@python3 tests/python/test_golden.py --flux-binary ./$(TARGET)

pngtest:
	@echo "Running PNG compression compare test..."
	@$(CC) $(CFLAGS_BASE) src/tools/png_compare.c src/core/flux_image.c -lm -o /tmp/flux_png_compare
	@/tmp/flux_png_compare images/woman_with_sunglasses.png images/woman_with_sunglasses_compressed2.png
	@/tmp/flux_png_compare images/cat_uncompressed.png images/cat_compressed.png
	@rm -f /tmp/flux_png_compare
	@echo "PNG TEST PASSED"

install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include/flux
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 include/flux/flux.h /usr/local/include/flux/
	install -m 644 include/flux/flux_kernels.h /usr/local/include/flux/

clean:
	rm -f $(OBJS) $(CORE_SRCS:.c=.mps.o) $(CLI_SRCS:.c=.mps.o) src/app/main.mps.o flux_metal.o $(TARGET) $(LIB)
	rm -f $(CUDA_C_OBJS) $(CUDA_OBJS)
	rm -f flux_shaders_source.h

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo ""
	@echo "Available backends for this platform:"
	@echo "  generic - Pure C (always available)"
ifeq ($(UNAME_S),Darwin)
	@echo "  blas    - Apple Accelerate"
ifeq ($(UNAME_M),arm64)
	@echo "  mps     - Metal GPU (recommended)"
endif
else
	@echo "  blas    - OpenBLAS (requires libopenblas-dev)"
endif
