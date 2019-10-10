#ifndef GIST_CUDA_CALC_ENTROPY_H
#define GIST_CUDA_CALC_ENTROPY_H

#include "ConstantsG.cuh"
#include <stdio.h>

class Dimensions {
public:
  int x_;
  int y_;
  int z_;
  int size_;

  __host__ __device__
  Dimensions() {}

  __host__ __device__
  Dimensions(int x, int y, int z) {
    this->x_ = x;
    this->y_ = y;
    this->z_ = z;
    this->size_ = x * y * z;
  }
  
  __host__ __device__
  Dimensions &operator=(const Dimensions other) {
    this->x_ = other.x_;
    this->y_ = other.y_;
    this->z_ = other.z_;
    this->size_ = other.size_;
    return *this;
  }

};

class EntropyCalculator {
public:
  __host__ __device__ EntropyCalculator() {}
  __host__ __device__ EntropyCalculator(QuaternionG<float> *quaternions, float *coords, Dimensions gridsize, int *cumsum, float temp, float rho0, int nFrames) {
    this->quaternions = quaternions;
    this->coords = coords;
    this->griddims = gridsize;
    this->gridsize = gridsize.size_;
    this->cumSumAtoms = cumsum;
    this->temp = temp;
    this->rho0 = rho0;
    this->nFrames = nFrames;
    this->dTSorient = 0;
    this->dTSsix = 0;
    this->dTStrans = 0;
  }
  __device__ void calcOrientEntropy(int voxel) {
    int nwtotal = 0;
    int start = 0;
    int end = 0;
    if (voxel == 0) {
      end = this->cumSumAtoms[0];
      nwtotal = end;
    } else {
      start = this->cumSumAtoms[voxel - 1];
      end = this->cumSumAtoms[voxel];
      nwtotal = end - start;
    }
    if (nwtotal < 2) {
      this->dTSorient = 0.0;
      return;
    }
    float result = 0.0;
    for (int n0 = start; n0 < end; ++n0) {
      float NNr = ConstantsG::HUGEG;
      QuaternionG<float> quat1 = this->quaternions[n0];
      for (int n1 = start; n1 < end; ++n1) {
        if (n1 == n0) {
          continue;
        }
        QuaternionG<float> quat2 = this->quaternions[n1];
  
        float rR = quat1.distance(quat2);
        if ( (rR < NNr) && (rR > ConstantsG::SMALL) ) {
          NNr = rR;
        }
      }
      if (NNr < ConstantsG::HUGEG && NNr > ConstantsG::SMALL) {
        result += log(NNr * NNr * NNr * nwtotal / (3.0 * ConstantsG::TWOPI));
      }
    }
    this->dTSorient = ConstantsG::GASK_KCAL * this->temp * (result / (float)nwtotal + ConstantsG::EULER_MASC);
  }


  __device__ void calcTransEntropy(int voxel) {
    int step[3] = {this->griddims.z_ * this->griddims.y_, this->griddims.z_, 1};
    int x = voxel / step[0];
    int y = (voxel / step[1]) % this->griddims.y_;
    int z = voxel % step[1];
    float framesRho = this->nFrames * this->rho0;
    
    int nwtotal = 0;
    int start = 0;
    int end = 0;
    if (voxel == 0) {
      end = this->cumSumAtoms[0];
      nwtotal = end;
    } else {
      start = this->cumSumAtoms[voxel - 1];
      end = this->cumSumAtoms[voxel];
      nwtotal = end - start;
    }
    if (nwtotal < 1) {
      this->dTStrans = 0.0;
      return;
    }
    float result = 0.0;
    float resultSix = 0.0;

    for (int n0 = start; n0 < end; ++n0) {
      this->NNd = ConstantsG::HUGEG;
      this->NNs = ConstantsG::HUGEG;
      // Push further ahead!!
      if ( !((x <= 0                    ) || (y <= 0                ) || (z <= 0                ) ||
            (x >= this->griddims.x_ - 1) || (y >= this->griddims.y_ - 1) || (z >= this->griddims.z_ - 1) )) {
        
        for (int dim_1 = -1; dim_1 < 2; ++dim_1) {
          for (int dim_2 = -1; dim_2 < 2; ++dim_2) {
            for (int dim_3 = -1; dim_3 < 2; ++dim_3) {
              int voxel2 = voxel + dim_1 * step[0] + dim_2 * step[1] + dim_3 * step[2];
              this->calcTransEntropyDist(voxel, voxel2, n0);
            }
          }
        } 
      } else {
        this->dTStrans = 0;
        return;
      }
      float NNd_s = sqrt(this->NNd);
      if (NNd_s < 3 && NNd_s > ConstantsG::SMALL) {
        result += log(NNd_s * this->NNd * ConstantsG::FOURPI * framesRho / 3.0);
        resultSix += log(this->NNs * this->NNs * this->NNs * framesRho * ConstantsG::PI / 48.0);
      }
    }
    this->dTStrans = ConstantsG::GASK_KCAL * this->temp * (result / nwtotal + ConstantsG::EULER_MASC);
    this->dTSsix = ConstantsG::GASK_KCAL * this->temp * (resultSix / nwtotal + ConstantsG::EULER_MASC);
  }

  __device__ void calcTransEntropyDist(int voxel1, int voxel2, int water1) {
    int start2 = 0;
    int end2 = 0;
    float coord1x = this->coords[3 * water1    ];
    float coord1y = this->coords[3 * water1 + 1];
    float coord1z = this->coords[3 * water1 + 2];
    if (voxel2 == 0) {
      end2 = this->cumSumAtoms[0];
    } else {
      start2 = this->cumSumAtoms[voxel2 - 1];
      end2 = this->cumSumAtoms[voxel2];
    }
  
    for (int water2 = start2; water2 < end2; ++water2) {
      if (voxel1 == voxel2 && water1 == water2) {
        continue;
      }
      float x = coord1x - this->coords[3 * water2    ];
      float y = coord1y - this->coords[3 * water2 + 1];
      float z = coord1z - this->coords[3 * water2 + 2];
      float dd = x * x + y * y + z * z;
      if (dd > ConstantsG::SMALL && dd < this->NNd) {
        this->NNd = dd;
      }
      float rR = this->quaternions[water1].distance(this->quaternions[water2]);
      double ds = rR * rR + dd;
      if (ds < this->NNs && ds > ConstantsG::SMALL) {
        this->NNs = ds;
      }
    }
  }

  // Number quaternions = number coords / 3 = cumSumAtom[gridsize - 1]
  QuaternionG<float> *quaternions;
  float *coords;
  Dimensions griddims;
  int gridsize;
  int *cumSumAtoms;
  float temp;
  float rho0;
  float NNs;
  float NNd;
  int nFrames;
  float dTStrans;
  float dTSorient;
  float dTSsix;
};



#endif