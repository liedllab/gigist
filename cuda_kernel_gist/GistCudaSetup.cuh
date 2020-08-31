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

struct EnergyReturn {
  std::vector<float> eww;
  std::vector<float> esw;
};

// Function definitions
void allocateCuda(void **, int);
void copyMemoryToDevice(void *, void *, int);
void copyMemoryToDeviceStruct(float *, int *, bool *, int *, int, void **, float *, float *, int, void **);
void freeCuda(void *);
EnergyReturn doActionCudaEnergy(const double *coords, int *NBindex_c, int ntypes, void *parameter, void *molecule_c,
                            int boxinfo, float *recip_o_box, float *ucell, int maxAtoms, int headAtomType, 
                            float neighbourCut2, int *result_o, int *result_n, float *result_w_c, float *result_s_c,
                            int *result_O_c, int *result_N_c, bool doorder);
std::vector<std::vector<float> > doActionCudaEntropy(std::vector<std::vector<Vec3> >, int, int, int, std::vector<std::vector<Quaternion<float> > >, float, float, int);

#endif
