#ifndef GIST_CUDA_SETUP_CUH
#define GIST_CUDA_SETUP_CUH

#include "../Quaternion.h"
#include "../Vec3.h"

#include <exception>
#include <vector>
#include <iostream>


// Exception classes
class CudaException : public std::exception {
public:
  CudaException() {
  }
};

// Function definitions
void allocateCuda(void **, int);
void copyMemoryToDevice(void *, void *, int);
void copyMemoryToDeviceStruct(float *, int *, bool *, int *, int, void **, float *, float *, int, void **);
void freeCuda(void *);
std::vector<std::vector<float> > doActionCudaEnergy(const double *, int *, int , void *, void *,
                            int , float *, float *, int , float *, 
                            float *, int, float, int *, int *, float *, float *,
                            int *, int *, bool);
std::vector<std::vector<float> > doActionCudaEntropy(std::vector<std::vector<Vec3> >, int, int, int, std::vector<std::vector<Quaternion<float> >>, float, float, int);

#endif
