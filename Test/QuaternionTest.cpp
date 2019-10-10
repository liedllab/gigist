#include "../Quaternion.h"
#include "gtest/gtest.h"


// Test for equality check
TEST(QuaternionTest, QuaternionTestEquals) {
  Quaternion<float> quat1(1, 1, 1, 1);
  Quaternion<float> quat2(1, 1, 1, 1);
  ASSERT_TRUE(quat1 == quat2);
}

// Test the access via paranthesis
TEST(QuaternionTest, QuaternionTestAccess) {
  Quaternion<float> quat(1, 2, 3, 4);
  ASSERT_FLOAT_EQ(quat[0], 1);
  ASSERT_FLOAT_EQ(quat[1], 2);
  ASSERT_FLOAT_EQ(quat[2], 3);
  ASSERT_FLOAT_EQ(quat[3], 4);
  ASSERT_ANY_THROW(quat[4]);
}

// Test the multiplication
TEST(QuaternionTest, QuaternionTestMultiplication) {
  Quaternion<float> quat(1,1,1,1);
  Quaternion<float> quat2(1,1,1,1);
  Quaternion<float> tester = quat * quat2;
  Quaternion<float> correct(-2, 2, 2, 2);
  ASSERT_TRUE(tester == correct);
}

// Test inversion of the quaternion
TEST(QuaternionTest, QuaternionTestInvert) {
  Quaternion<float> tester(1,1,1,1);
  tester = tester.invert();
  Quaternion<float> correct(1, -1, -1, -1);
  ASSERT_TRUE(tester == correct);
}

// Test the distance calculation for two different quaternions.
TEST(QuaternionTest, QuaternionTestDistance) {
  // Test correct calculation of same to same.
  Quaternion<float> quat(0.5, 0.5, 0.5, 0.5);
  Quaternion<float> quat2(0.5, 0.5, 0.5, 0.5);
  ASSERT_FLOAT_EQ(quat.distance(quat2), 0);
  // Test correct calculation of distance same to negative (0 distance)
  // Compare with test for quaternion rotation, why this is the case.
  quat2 = Quaternion<float>(-0.5, -0.5, -0.5, -0.5);
  ASSERT_FLOAT_EQ(quat.distance(quat2), 0);
  // Test correct calculation of a random (more or less) distance.
  quat2 = Quaternion<float>(1, 0, 0, 0);
  ASSERT_FLOAT_EQ(quat.distance(quat2), 2.094395102);
}

// Check the constructor.
TEST(QuaternionTest, QuaternionTestConstructor) {
  // Just checks, nothing can ever happen here
  ASSERT_NO_THROW(Quaternion<float>(0.5, 0.5, 0.5, 0.5));
  // Check for correct calculation of the Quaternion from
  // vectors.
  Quaternion<float> quat(0.5, 0.5, 0.5, 0.5);
  Vec3 vec1(0, 1, 0);
  Vec3 vec2(0, 0, 1);
  Quaternion<float> quat2(vec1, vec2);
  ASSERT_EQ(quat2, quat);
}

// Check Rotations.
TEST(QuaternionTest, QuaternionTestRotation) {
  // Check identity rotation
  Quaternion<float> rotator(1, 0, 0, 0);
  Vec3 vector(0,1,0);
  ASSERT_EQ(rotator.rotate(vector), vector);
  // Check rotation around X
  Quaternion<float> rot2(1/sqrt(2), 1/sqrt(2), 0, 0);
  ASSERT_EQ(rot2.rotate(vector), Vec3(0,0,1)); 
  // Check inverted rotation, to negative values
  rot2 = rot2.invert();
  ASSERT_EQ(rot2.rotate(vector), Vec3(0,0,-1));
  // Check rotation of (0, 1, 0), (0, 0, 1), (1, 0, 0)
  // to (1, 0, 0), (0, 1, 0), (0, 0, 1) (Lab Coordinates)
  rotator = Quaternion<float>(0.5, 0.5, 0.5, 0.5);
  ASSERT_EQ(rotator.invert().rotate(Vec3(0, 1, 0)), Vec3(1, 0, 0));
  ASSERT_EQ(rotator.invert().rotate(Vec3(0, 0, 1)), Vec3(0, 1, 0));
  ASSERT_EQ(rotator.invert().rotate(Vec3(1, 0, 0)), Vec3(0, 0, 1));
  // Check if negative quaternion is same rotation as positive.
  rotator = Quaternion<float>(-0.5, -0.5, -0.5, -0.5);
  ASSERT_EQ(rotator.invert().rotate(Vec3(0, 1, 0)), Vec3(1, 0, 0));
  ASSERT_EQ(rotator.invert().rotate(Vec3(0, 0, 1)), Vec3(0, 1, 0));
  ASSERT_EQ(rotator.invert().rotate(Vec3(1, 0, 0)), Vec3(0, 0, 1));
}