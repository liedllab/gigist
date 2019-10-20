#include "GistCudaSetup.cuh"
#include "GistCudaCalc.cuh"
#include "EntropyCalculator.cuh"
#include <iostream>


/**
 * Allocate memory on the GPU.
 * @parameter array: The pointer to the array, which will be allocated on the GPU.
 * @parameter size: An integer giving the size of the array, which will be allocated.
 * @throws: CudaException if a problem occurs.
 */
__host__
void allocateCuda(void **array, int size) {
  // Check if the array is actually free, if not, it will be freed 
  // (fun fact: checking is not necessary, one could also simply free the memory).
  if ((*array) != NULL) {
    cudaFree(*array);
  }
  // If something goes wrong, throw exception
  if (cudaMalloc(array, size) != cudaSuccess) {
    throw CudaException();
  }
}

/**
 * Copy memory from the CPU to the GPU.
 * @parameter array: The array from which the values shall be copied.
 * @parameter array_c: The array on the device, to which the values shall be copied.
 * @parameter size: The size of the stuff which will be copied.
 * @throws: CudaException if something goes wrong.
 */
__host__
void copyMemoryToDevice(void *array, void *array_c, int size) {
  // If something goes wrong, throw exception
  // In this case only copying can go wrong.
  if (cudaMemcpy(array_c, array, size, cudaMemcpyHostToDevice) != cudaSuccess) {
    throw CudaException();
  }
}

/**
 * A simple helper function that copies a lot of stuff to the GPU (as structs).
 * @parameter charge: An array holding the charges for the different atoms.
 * @parameter atomtype: An array holding the integers for the atom types of the different atoms.
 * @parameter solvent: An array of boolean values, holding the information whether a certain atom is solvent or solute.
 * @parameter atomNumber: The total number of atoms.
 * @parameter atomProps_c: A pointer to an array on the GPU, which will hold the atom properties.
 * @parameter ljA: An array holding the lennard-jones parameter A for each atom type pair.
 * @parameter ljB: An array holding the lennard-jones parameter B for each atom type pair.
 * @parameter length: The length of the two aforementioned arrays (ljA & ljB).
 * @parameter lJparams_c: A pointer to an array on the GPU, which will hold the lj parameters.
 * @throws: CudaException if something bad happens.
 */
__host__
void copyMemoryToDeviceStruct(float *charge, int *atomtype, bool *solvent, int *molecule, int atomNumber, void **atomProps_c,
                              float *ljA, float *ljB, int length, void **lJparams_c) {
  // Check if the two arrays are free. Again, this could be removed (but will stay!)
  if ((*atomProps_c) != NULL) {
    cudaFree(*atomProps_c);
  }
  if ((*lJparams_c) != NULL) {
    cudaFree(*lJparams_c);
  }
  // Allocate the necessary memory on the GPU.
  if (cudaMalloc(atomProps_c, atomNumber * sizeof(AtomProperties)) != cudaSuccess) {
    throw CudaException();
  }
  if (cudaMalloc(lJparams_c, length * sizeof(ParamsLJ)) != cudaSuccess) {
    throw CudaException();
  }

  // Create an array for the lennard-jones parameters.
  ParamsLJ *ljp = (ParamsLJ *) malloc (length * sizeof(ParamsLJ));
  // Add the lennard-jones parameters to the array.
  for (int i = 0; i < length; ++i) {
    ljp[i] = ParamsLJ(ljA[i], ljB[i]);
  }

  // Create an array for the atom properties.
  AtomProperties *array = (AtomProperties *)malloc(atomNumber * sizeof(AtomProperties));
  // Add the properties into the array.
  for (int i = 0; i < atomNumber; ++i) {
    array[i] = AtomProperties(charge[i], atomtype[i], solvent[i], molecule[i]);
  }
  // Copy the memory from the host to the device.
  if (cudaMemcpy((*atomProps_c), array, atomNumber * sizeof(AtomProperties), cudaMemcpyHostToDevice) != cudaSuccess) {
    throw CudaException();
  }
  if (cudaMemcpy((*lJparams_c), ljp, length * sizeof(ParamsLJ), cudaMemcpyHostToDevice) != cudaSuccess) {
    throw CudaException();
  }

  // Free the two arrays (so that no memory leak occurs).
  free(ljp);
  free(array);
}

/**
 * Free an array.
 * @parameter array: The array you want to free.
 */
__host__
void freeCuda(void *array) {
  cudaFree(array);
}


// This is coded C-like, but uses exceptions.
/**
 * This starts the cuda kernel, thus it is actually a quite long function.
 */
__host__
std::vector<std::vector<float> > doActionCudaEnergy(const double *coords, int *NBindex_c, int ntypes, void *parameter, void *molecule_c,
                            int boxinfo, float *recip_o_box, float *ucell, int maxAtoms, float *min_c, float *max_c, int headAtomType, 
                            float neighbourCut2, int *result_o, int *result_n, float *result_w_c, float *result_s_c,
                            int *result_O_c, int *result_N_c, bool doorder) {
  Test *coords_c   = NULL;
  float *recip_b_c  = NULL;
  float *ucell_c    = NULL;
  
  

  float *result_A = (float *) calloc(maxAtoms, sizeof(float));
  float *result_s = (float *) calloc(maxAtoms, sizeof(float));
  // TODO: Fix this, test is actually a quite bad name here!
  Test *coord_array = (Test *) calloc(maxAtoms, sizeof(Test));
  
  // Casting
  AtomProperties *sender = (AtomProperties *) molecule_c;
  ParamsLJ *lennardJonesParams = (ParamsLJ *) parameter;
  
  // Create Boxinfo and Unit cell. This is actually very important for the speed (otherwise
  // there would be LOTS of access to non-local variables).
  BoxInfo boxinf;
  if (boxinfo != 0) {
    boxinf = BoxInfo(recip_o_box, boxinfo);
  }
  UnitCell ucellN;
  if (boxinfo == 2) {
    ucellN = UnitCell(ucell);
  }
  
  // Add the coordinates to the array.
  // TODO: Fix Test here also!
  for (int i = 0; i < maxAtoms; ++i) {
    coord_array[i] = Test(&coords[i * 3]);
  }

  // vectors that will return the necessary information.
  std::vector<std::vector<float> > result;
  std::vector<float> result_esw;
  std::vector<float> result_eww;

  // Allocate space on the GPU
  if (cudaMalloc(&coords_c, maxAtoms * sizeof(Test)) != cudaSuccess) {
    free(result_A); free(result_s); free(coord_array);
    throw CudaException();
  }


  // Copy the data to the GPU
  if (cudaMemcpy(coords_c, coord_array, maxAtoms * sizeof(Test), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(coords_c); cudaFree(recip_b_c); cudaFree(ucell_c);
    free(result_A); free(result_s); free(coord_array);
    throw CudaException();
  }
  if (cudaMemcpy(result_w_c, result_A, maxAtoms * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(coords_c); cudaFree(recip_b_c); cudaFree(ucell_c);
    free(result_A); free(result_s); free(coord_array);
    throw CudaException();
  }
  if (cudaMemcpy(result_s_c, result_s, maxAtoms * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(coords_c); cudaFree(recip_b_c); cudaFree(ucell_c);
    free(result_A); free(result_s); free(coord_array);
    throw CudaException();
  }

  // If the doorder calculation is used, it needs to calculate everything differently, so the slow version is used
  // (this is about 10% slower).
  if (doorder) {
    cudaCalcEnergySlow<<< (maxAtoms + SLOW_BLOCKSIZE) / SLOW_BLOCKSIZE, SLOW_BLOCKSIZE >>> (coords_c, NBindex_c, ntypes, lennardJonesParams, sender,
                                                                                            boxinf, ucellN, maxAtoms, result_w_c, result_s_c, min_c, max_c,
                                                                                            headAtomType, neighbourCut2, result_O_c, result_N_c);
  } else {
    // Uses a 2D array, which is nice for memory access.
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 numBlocks((maxAtoms + threadsPerBlock.x) / threadsPerBlock.x, (maxAtoms + threadsPerBlock.y) / threadsPerBlock.y);
    // The actual call of the device function
    cudaCalcEnergy<<<numBlocks, threadsPerBlock>>> (coords_c, NBindex_c, ntypes, lennardJonesParams, sender,
                                                                      boxinf, ucellN, maxAtoms, result_w_c, result_s_c, min_c, max_c,
                                                                      headAtomType, neighbourCut2, result_O_c, result_N_c);
    // Check if there was an error.
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
      printf("returned %s\n", cudaGetErrorString(cudaError));
    }
  }
  // Return the results of the calculation to the main memory
  if (cudaMemcpy(result_A, result_w_c, maxAtoms * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(coords_c); cudaFree(recip_b_c); cudaFree(ucell_c);
    free(result_A); free(result_s); free(coord_array);
    throw CudaException();
  }  
  

  if (cudaMemcpy(result_s, result_s_c, maxAtoms * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(coords_c); cudaFree(recip_b_c); cudaFree(ucell_c);
    free(result_A); free(result_s); free(coord_array);
    throw CudaException();
  }


  
  if (cudaMemcpy(result_o, result_O_c, maxAtoms * 4 * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(coords_c); cudaFree(recip_b_c); cudaFree(ucell_c);
    free(result_A); free(result_s); free(coord_array);
    throw CudaException();
  }
  
  if (cudaMemcpy(result_n, result_N_c, maxAtoms * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(coords_c); cudaFree(recip_b_c); cudaFree(ucell_c);
    free(result_A); free(result_s); free(coord_array);
    throw CudaException();
  }

  for (int i = 0; i < maxAtoms; ++i) {
    result_eww.push_back(result_A[i]);
    result_esw.push_back(result_s[i]);
  }

  result.push_back(result_eww);
  result.push_back(result_esw);

  // Free everything used in here.
  cudaFree(coords_c); cudaFree(recip_b_c); cudaFree(ucell_c);
  free(result_A); free(result_s); free(coord_array);
  
  return result;
}

#ifdef DEBUG_GIST_CUDA
// Not necessary
__host__
std::vector<Quaternion<float> > shoveQuaternionsTest(std::vector<Quaternion<float> > quats) {
  QuaternionG<float> *quats_c = NULL;
  float *ret_c = NULL;
  std::vector<Quaternion<float> > ret;
  float *ret_f = new float[quats.size() * 4];
  QuaternionG<float> *quats_f = new QuaternionG<float>[quats.size()];
  for (int i = 0; i < quats.size(); ++i) {
    quats_f[i] = quats.at(i);
  }
  if (cudaMalloc(&quats_c, quats.size() * sizeof(QuaternionG<float>)) != cudaSuccess) {
    delete quats_f; delete ret_f;
    throw CudaException();
  }
  if (cudaMalloc(&ret_c, quats.size() * 4 * sizeof(float)) != cudaSuccess) {
    cudaFree(quats_c);
    delete quats_f; delete ret_f;
    throw CudaException();
  }

  if (cudaMemcpy(quats_c, quats_f, quats.size() * sizeof(QuaternionG<float>), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(quats_c); cudaFree(ret_c);
    delete quats_f; delete ret_f;
    throw CudaException();
  }

  shoveQuaternions<<< (quats.size() + BLOCKSIZE) / BLOCKSIZE, BLOCKSIZE >>> (quats_c, quats.size(), ret_c);

  if (cudaMemcpy(ret_f, ret_c, quats.size() * 4 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(quats_c); cudaFree(ret_c);
    delete quats_f; delete ret_f;
    throw CudaException();
  }

  for (int i = 0; i < quats.size(); ++i) {
    ret.push_back(Quaternion<float>(ret_f[i * 4], ret_f[i * 4 + 1], ret_f[i * 4 + 2], ret_f[i * 4 + 3]));
  }

  cudaFree(quats_c); cudaFree(ret_c);
  delete quats_f; delete ret_f;

  

  return ret;
}
#endif

/**
 * Calculates the entropy on the GPU (this is not really necessary and does not lead to a significant speed up).
 * @parameter coords: The coordinates of the different water molecules.
 * @parameter x: The number of grid voxels in the x direction.
 * @parameter y: The number of grid voxels in the y direction.
 * @parameter z: The number of grid voxels in the z direction.
 * @parameter quats: A vector object holding all the quaternions.
 * @parameter temp: The temperature.
 * @parameter rho0: The reference density.
 * @parameter nFrames: The total number of frames.
 * @return: A vector holding the values for dTStrans, dTSorient and dTSsix.
 * @throws: A CudaException on error.
 */
std::vector<std::vector<float> > doActionCudaEntropy(std::vector<std::vector<Vec3> > coords, int x, int y, int z, std::vector<std::vector<Quaternion<float> > > quats, float temp, float rho0, int nFrames) {
  
  // For the CPU
  // Input (from previous calculations)
  std::vector<QuaternionG<float> > quatsF;
  std::vector<float> coordsF;
  std::vector<int> cumSumAtoms;
  // Results
  float *resultTStrans  = new float[quats.size()];
  float *resultTSorient = new float[quats.size()];
  float *resultTSsix    = new float[quats.size()];

  // For the GPU
  // Input (from previous calculations)
  Dimensions dims            = Dimensions(x, y, z);
  float *coordsG             = NULL;
  QuaternionG<float> *quatsG = NULL;
  int *cumSumAtomsG          = NULL;
  // Results
  float *resultTStransG       = NULL;
  float *resultTSorientG      = NULL;
  float *resultTSsixG         = NULL;
  
  int sum = 0;
  for (int i = 0 ; i < quats.size(); ++i) {
    sum += quats.at(i).size();
    cumSumAtoms.push_back(sum);
    for (int j = 0; j < quats.at(i).size(); ++j) {
      // quatsF always has the size of the number of the current molecule. 
      coordsF.push_back((float) (coords.at(i).at(j)[0]));
      coordsF.push_back((float) (coords.at(i).at(j)[1]));
      coordsF.push_back((float) (coords.at(i).at(j)[2]));
      quatsF.push_back(quats.at(i).at(j));
    }
  }



  cudaError_t err1 = cudaMalloc(&quatsG, quatsF.size() * sizeof(QuaternionG<float>));
  cudaError_t err2 = cudaMalloc(&coordsG, coordsF.size() * sizeof(float));
  cudaError_t err3 = cudaMalloc(&cumSumAtomsG, cumSumAtoms.size() * sizeof(int));
  cudaError_t err4 = cudaMalloc(&resultTStransG, quats.size() * sizeof(float));
  cudaError_t err5 = cudaMalloc(&resultTSorientG, quats.size() * sizeof(float));
  cudaError_t err6 = cudaMalloc(&resultTSsixG, quats.size() * sizeof(float));
  // Error Check
  if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess ||
      err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess) {
    cudaFree(quatsG);
    cudaFree(coordsG);
    cudaFree(cumSumAtomsG);
    cudaFree(resultTStransG);
    cudaFree(resultTSorientG);
    cudaFree(resultTSsixG);
    delete[] resultTStrans;
    delete[] resultTSorient;
    delete[] resultTSsix;
    throw CudaException();
  }


  err1 = cudaMemcpy(quatsG, &(quatsF[0]), quatsF.size() * sizeof(QuaternionG<float>), cudaMemcpyHostToDevice);
  err2 = cudaMemcpy(coordsG, &(coordsF[0]), coordsF.size() * sizeof(float), cudaMemcpyHostToDevice);
  err3 = cudaMemcpy(cumSumAtomsG, &(cumSumAtoms[0]), cumSumAtoms.size() * sizeof(int), cudaMemcpyHostToDevice);

  
  // Error Check
  if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
    cudaFree(quatsG);
    cudaFree(coordsG);
    cudaFree(cumSumAtomsG);
    cudaFree(resultTStransG);
    cudaFree(resultTSorientG);
    cudaFree(resultTSsixG);
    delete[] resultTStrans;
    delete[] resultTSorient;
    delete[] resultTSsix;
    throw CudaException();
  }

  EntropyCalculator entCalc = EntropyCalculator(quatsG, coordsG, dims, cumSumAtomsG, temp, rho0, nFrames);
  calculateEntropy<<<(quats.size() + SLOW_BLOCKSIZE) / SLOW_BLOCKSIZE, SLOW_BLOCKSIZE>>>(entCalc, resultTStransG, resultTSorientG, resultTSsixG);
  cudaError_t err7 = cudaGetLastError();

  // Error Check
  if (err7 != cudaSuccess) {
    cudaFree(quatsG);
    cudaFree(coordsG);
    cudaFree(cumSumAtomsG);
    cudaFree(resultTStransG);
    cudaFree(resultTSorientG);
    cudaFree(resultTSsixG);
    delete[] resultTStrans;
    delete[] resultTSorient;
    delete[] resultTSsix;
    throw CudaException();
  }

  // Copy back, use same errors as above for understandability.
  err4 = cudaMemcpy(resultTStrans, resultTStransG, quats.size() * sizeof(float), cudaMemcpyDeviceToHost);
  err5 = cudaMemcpy(resultTSorient, resultTSorientG, quats.size() * sizeof(float), cudaMemcpyDeviceToHost);
  err6 = cudaMemcpy(resultTSsix, resultTSsixG, quats.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // Don't need that anymore.
  cudaFree(quatsG);
  cudaFree(coordsG);
  cudaFree(cumSumAtomsG);
  cudaFree(resultTStransG);
  cudaFree(resultTSorientG);
  cudaFree(resultTSsixG);

  // Error Check
  if (err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess) {
    delete[] resultTStrans;
    delete[] resultTSorient;
    delete[] resultTSsix;
    throw CudaException();
  }

  std::vector<float> trans;
  std::vector<float> orient;
  std::vector<float> six;

  for (int i = 0; i < quats.size(); ++i) {
    trans.push_back(resultTStrans[i]);
    orient.push_back(resultTSorient[i]);
    six.push_back(resultTSsix[i]);
  }

  std::vector<std::vector<float> > ret;
  ret.push_back(trans);
  ret.push_back(orient);
  ret.push_back(six);

  delete[] resultTStrans;
  delete[] resultTSorient;
  delete[] resultTSsix;

  return ret;
}
