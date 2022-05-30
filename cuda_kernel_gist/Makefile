include ../../config.h

DEL_FILE      = /bin/rm -f
TARGET        = lib_cuda_gist.a
AR            = ar cqs

# General rules

%.o : %.cu
	$(NVCC) $(DBGFLAGS) -c $(NVCCFLAGS) -lineinfo -o $@ $<

# Source files

CUDA_SOURCES=GistCudaCalc.cu GistCudaSetup.cu

# Objects

OBJECTS=$(CUDA_SOURCES:.cu=.o)

# Specific rules

install: $(TARGET)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	-$(DEL_FILE) $(TARGET)
	$(AR) $(TARGET) $(OBJECTS)

clean:
	$(DEL_FILE) $(TARGET) *.o

uninstall: clean
