#ifndef GIST_CUDA_CALC_CUH
#define GIST_CUDA_CALC_CUH

#include "QuaternionG.cuh"
#include "EntropyCalculator.cuh"

#define HUGE_C 1e200
#define BLOCKSIZE 16
#define SLOW_BLOCKSIZE 512

class Coordinates_GPU {
public:
  float x;
  float y;
  float z;

  __host__ __device__
  Coordinates_GPU(): x(0), y(0), z(0) {}

  __host__ __device__
  Coordinates_GPU(const double *array) {
    this->x = array[0];
    this->y = array[1];
    this->z = array[2];
  }

  __host__ __device__
  Coordinates_GPU(const Coordinates_GPU &other) {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
  }

  __host__ __device__
  Coordinates_GPU &operator=(const Coordinates_GPU &other) {
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    return *this;
  }
  
};

class ParamsLJ{
public:
  float A;
  float B;

  __host__ __device__
  ParamsLJ(): A(0), B(0) {}

  __host__ __device__
  ParamsLJ(float *arr) {
    this->A = arr[0];
    this->B = arr[1];
  }

  __host__ __device__
  ParamsLJ(float A, float B) {
    this->A = A;
    this->B = B;
  }

  __host__ __device__
  ParamsLJ(const ParamsLJ &other) {
    this->A = other.A;
    this->B = other.B;
  }

  __host__ __device__
  ParamsLJ &operator=(const ParamsLJ &other) {
    this->A = other.A;
    this->B = other.B;
    return *this;
  }
};

class UnitCell {
public:
  float array[9];
  __host__ __device__
  UnitCell() {}

  __host__ __device__
  UnitCell(float *arr) {
    this->array[0] = arr[0];
    this->array[1] = arr[1];
    this->array[2] = arr[2];
    this->array[3] = arr[3];
    this->array[4] = arr[4];
    this->array[5] = arr[5];
    this->array[6] = arr[6];
    this->array[7] = arr[7];
    this->array[8] = arr[8];
  }

  __host__ __device__
  UnitCell(const UnitCell &other) {
    this->array[0] = other.array[0];
    this->array[1] = other.array[1];
    this->array[2] = other.array[2];
    this->array[3] = other.array[3];
    this->array[4] = other.array[4];
    this->array[5] = other.array[5];
    this->array[6] = other.array[6];
    this->array[7] = other.array[7];
    this->array[8] = other.array[8];
  }

  __host__ __device__
  UnitCell &operator=(const UnitCell &other){
    this->array[0] = other.array[0];
    this->array[1] = other.array[1];
    this->array[2] = other.array[2];
    this->array[3] = other.array[3];
    this->array[4] = other.array[4];
    this->array[5] = other.array[5];
    this->array[6] = other.array[6];
    this->array[7] = other.array[7];
    this->array[8] = other.array[8];
    return *this;
  }

  __host__ __device__
  float operator[](int idx) const {
    if (idx >= 0 && idx < 9) {
      return this->array[idx];
    }
    return 1;
  }

};

class BoxInfo {
public:
  float array[9];
  int boxinfo;
  __host__ __device__
  BoxInfo(): boxinfo(0) {}

  __host__ __device__
  BoxInfo(float *arr, int boxinfo) {
    this->array[0] = arr[0];
    this->array[1] = arr[1];
    this->array[2] = arr[2];
    this->array[3] = arr[3];
    this->array[4] = arr[4];
    this->array[5] = arr[5];
    this->array[6] = arr[6];
    this->array[7] = arr[7];
    this->array[8] = arr[8];
    this->boxinfo = boxinfo;
  }

  __host__ __device__
  BoxInfo(const BoxInfo &other) {
    this->array[0] = other.array[0];
    this->array[1] = other.array[1];
    this->array[2] = other.array[2];
    this->array[3] = other.array[3];
    this->array[4] = other.array[4];
    this->array[5] = other.array[5];
    this->array[6] = other.array[6];
    this->array[7] = other.array[7];
    this->array[8] = other.array[8];
    this->boxinfo = other.boxinfo;
  }

  __host__ __device__
  BoxInfo &operator=(const BoxInfo &other){
    this->array[0] = other.array[0];
    this->array[1] = other.array[1];
    this->array[2] = other.array[2];
    this->array[3] = other.array[3];
    this->array[4] = other.array[4];
    this->array[5] = other.array[5];
    this->array[6] = other.array[6];
    this->array[7] = other.array[7];
    this->array[8] = other.array[8];
    this->boxinfo = other.boxinfo;
    return *this;
  }

  __host__ __device__
  float operator[](int idx) const {
    if (idx < 9 && idx >= 0) {
      return this->array[idx];
    }
    return 0;
  }

};

class AtomProperties {
public:
  float charge;
  int atomType;
  bool solvent;
  int molecule;

  __host__ __device__
  AtomProperties() {}

  __host__ __device__
  AtomProperties(float charge, int atomType, bool solvent, int molecule) {
    this->charge = charge;
    this->atomType = atomType;
    this->solvent = solvent;
    this->molecule = molecule;
  }

  __host__ __device__
  AtomProperties(const AtomProperties &other) {
    this->charge = other.charge;
    this->atomType = other.atomType;
    this->solvent = other.solvent;
    this->molecule = other.molecule;
  }

  __host__  __device__
  AtomProperties &operator=(const AtomProperties &other) {
    this->charge = other.charge;
    this->atomType = other.atomType;
    this->solvent = other.solvent;
    this->molecule = other.molecule;
    return *this;
  }
};


// Device functions
__device__ float dist2_imageOrtho(float *, float *, const BoxInfo &);
__device__ void scalarProd(float* , const BoxInfo & , float *);
__device__ float dist2_imageNonOrtho(float *, float *, const BoxInfo &, const UnitCell &);
__device__ float calcIfDistIsSmaller(float *, float *, int , int , int , const UnitCell &, float );
__device__ float dist2_imageNonOrthoRecip(float * , float * , const UnitCell &);
__device__ float dist2_imageNonOrthoRecipTest(float * , float * , const UnitCell );
__device__ float dist2_noImage(float *, float *);
__device__ float calcTotalEnergy(float , float , 
                            float , float , float);
__device__ float calcVdWEnergy(float , float , float );
__device__ float calcElectrostaticEnergy(float, float, float);
__device__ int getLJIndex(int , int , int *, int );
__device__ ParamsLJ getLJParam(int , int , int *, int , ParamsLJ *);
__device__ bool isOnGrid(float *, float *, float *);
__device__ float calcDist(float *, float *, const BoxInfo &, const UnitCell &);

// Global functions
__global__ void cudaCalcEnergy    (Coordinates_GPU *, int *, int, ParamsLJ *, AtomProperties *, BoxInfo, UnitCell, int, float *, float *, float *, float *, int, float, int *, int *);
__global__ void cudaCalcEnergySlow(Coordinates_GPU *, int *, int, ParamsLJ *, AtomProperties *, BoxInfo, UnitCell, int, float *, float *,	float *, float *, int, float, int *, int *);
__global__ void calculateEntropy(EntropyCalculator entCalc, float *, float *, float *);
#endif