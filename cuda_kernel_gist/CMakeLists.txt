set(CPPTRAJ_CUDA_SOURCES GistCudaCalc.cu GistCudaSetup.cu)

cuda_add_library(cpptraj_cuda_routines ${CPPTRAJ_CUDA_SOURCES})

install_libraries(cpptraj_cuda_routines)
