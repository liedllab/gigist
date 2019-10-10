#include "gtest/gtest.h"

// Testing of CUDA code was done using a dedicated main function to test all different
// functions called on the GPU. However, as I did a lot of different changes to this code,
// I also changed a few things that would concern the tests and did not yet update the tests.
// Thus, the tests for the GPU code are not supplied here yet. And will be different to the
// Unit tests here. Also the functions from Action_GIGIST.cpp were tested in a main function
// there.

int main (int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
