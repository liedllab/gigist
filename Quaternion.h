#ifndef QUATERNION_H
#define QUATERNION_H

#include "ExceptionsGIST.h"
#include <math.h>
#ifndef TESTS
#include "Vec3.h"
#include "Constants.h"
#endif

#ifdef TESTS
#define SMALL 0.00001
class Vec3 {
public:
  double array[3];
  Vec3() {
    this->array[0] = 0;
    this->array[1] = 0;
    this->array[2] = 0;
  }
  Vec3(double x, double y, double z) {
    this->array[0] = x;
    this->array[1] = y;
    this->array[2] = z;
  }
  double operator[](int idx) {
    return this->array[idx];
  }
  void Normalize() {
    double norm = sqrt(this->array[0] * this->array[0] + this->array[1] * this->array[1] + this->array[2] * this->array[2]);
    this->array[0] /= norm;
    this->array[1] /= norm;
    this->array[2] /= norm; 
  }

  Vec3 Cross(Vec3 other) {
    double x = this->array[1] * other[2] - this->array[2] * other[1];
    double y = this->array[2] * other[0] - this->array[0] * other[2];
    double z = this->array[0] * other[1] - this->array[1] * other[0];
    return Vec3(x, y, z);
  }

  bool operator==(const Vec3 other) const {
    double x = fabs(this->array[0] - other.array[0]);
    double y = fabs(this->array[1] - other.array[1]);
    double z = fabs(this->array[2] - other.array[2]);
    return x < SMALL && y < SMALL && z < SMALL;
  }

};
#endif

/**
 * The Quaternion class to compare rotation angles.
 */
template <class T>
class Quaternion {
public:

	Quaternion<T>() {
		this->w_ = 0.0;
    this->x_ = 0.0;
    this->y_ = 0.0;
    this->z_ = 0.0;
	}
  /**
   * Constructor from a given set of quaternion coordinates.
   * @argument w: From the Quaternion theory.
   * @argument x: From the Quaternion theory.
   * @argument y: From the Quaternion theory.
   * @argument z: From the Quaternion theory.
   */
  Quaternion<T>(T w, T x, T y, T z) {
    this->w_ = w;
    this->x_ = x;
    this->y_ = y;
    this->z_ = z;
  }

  /**
   * Create Quaternion from two vectors. Be aware that this would rotate
   * the lab cooordinate system:
   * | 1 0 0 |
   * | 0 1 0 |
   * | 0 0 1 |
   * onto the two vectors. So to create the rotation of the vectors onto
   * the lab coordinate system, the quaternion has to be inverted.
   * @argument X: The X-Vector, that will be aligned with the X-Axis.
   * @argument V2: The second vector, this is not necessarily the Y-Axis.
   *               But will be orthogonal to the Z axis and in the X-Y plane.
   */
  Quaternion<T>(Vec3 X, Vec3 V2) {
    X.Normalize();
    Vec3 Z = X.Cross(V2);
    Z.Normalize();
    Vec3 Y = Z.Cross(X);
    Y.Normalize();
    // Create the 3x3 rotation matrix
    T m11 = X[0]; T m12 = Y[0]; T m13 = Z[0];
    T m21 = X[1]; T m22 = Y[1]; T m23 = Z[1];
    T m31 = X[2]; T m32 = Y[2]; T m33 = Z[2];

    // Calculate the trace of the rotation matrix
    T trace = m11 + m22 + m33;
    T s = 0;

    // Build the quaternion according to the rotation matrix.
    if (trace > 0) {

      s = 0.5 / sqrt( trace + 1 );

      this->w_ = 0.25 / s;
      this->x_ = (m32 - m23) * s;
      this->y_ = (m13 - m31) * s;
      this->z_ = (m21 - m12) * s;

    } else if (m11 > m22 && m11 > m33) {

      s = 2 * sqrt( 1.0 + m11 - m22 - m33);

      this->w_ = ( m32 - m23 ) / s;
      this->x_ = 0.25 * s;
      this->y_ = ( m12 + m21 ) / s;
      this->z_ = ( m13 + m31 ) / s;

    } else if (m22 > m33) {

      s = 2.0 * sqrt(1.0 + m22 - m11 - m33);

      this->w_ = ( m13 - m31 ) / s;
      this->x_ = ( m12 + m21 ) / s;
      this->y_ = 0.25 * s;
      this->z_ = ( m23 + m32 ) / s;

    } else {

      s = 2.0 * sqrt(1.0 + m33 - m11 - m22);

      this->w_ = ( m21 - m12 ) / s;
      this->x_ = ( m13 + m31 ) / s;
      this->y_ = ( m23 + m32 ) / s;
      this->z_ = 0.25 * s;

    }
  }

  /**
   * Constructor based on:
   * w = cos(theta / 2)
   * x = x1 * sin(theta / 2)
   * y = y1 * sin(theta / 2)
   * z = z1 * sin(theta / 2)
   * @argument angle: The angle for the rotation.
   * @argument turnVec: The vector around which the rotation will occur.
   */
  Quaternion<T>(T angle, Vec3 turnVec) {
    this->w_ = cos(angle / 2.0);
    T sinAngle = sin(angle / 2.0);
    this->x_ = turnVec[0] * sinAngle;
    this->y_ = turnVec[1] * sinAngle;
    this->z_ = turnVec[2] * sinAngle;
  }

  /**
   * The assignment operator.
   * @argument other: The Quaternion to be assigned.
   * @return: The updated Quaternion.
   */
  Quaternion<T> &operator=(Quaternion<T> other) {
    this->w_ = (T) other.W();
    this->x_ = (T) other.X();
    this->y_ = (T) other.Y();
    this->z_ = (T) other.Z();
    return *this;
  }

  /**
   * Bracket operator.
   * @argument idx: The index to access;
   * @return: The element stored at index, starting with w.
   */
  T operator[](int idx) {
    switch (idx) {
      case 0:
        return w_;
      case 1:
        return x_;
      case 2:
        return y_;
      case 3:
        return z_;
      default:
        throw IndexOutOfRangeException();
    }
  }

  /**
   * Multiply two different Quaternions.
   * @argument other: The second quaternion.
   * @return: This, updated by the multiplication.
   */
  Quaternion<T> &operator*=(Quaternion<T> other) {
    T w = this->w_;
    T x = this->x_;
    T y = this->y_;
    T z = this->z_;
    this->w_ = w * other.W() - x * other.X() - y * other.Y() - z * other.Z();
    this->x_ = w * other.X() + x * other.W() + y * other.Z() - z * other.Y();
    this->y_ = w * other.Y() - x * other.Z() + y * other.W() + z * other.X();
    this->z_ = w * other.Z() + x * other.Y() - y * other.X() + z * other.W();
    return *this;
  }

  /**
   * Multiply two different Quaternions.
   * @argument other: The second quaternion.
   * @return: A new quaternion holding the result of the multiplication.
   */
  Quaternion<T> operator*(Quaternion<T> other) {
    T w = this->w_ * other.W() - this->x_ * other.X() - this->y_ * other.Y() - this->z_ * other.Z();
    T x = this->w_ * other.X() + this->x_ * other.W() + this->y_ * other.Z() - this->z_ * other.Y();
    T y = this->w_ * other.Y() - this->x_ * other.Z() + this->y_ * other.W() + this->z_ * other.X();
    T z = this->w_ * other.Z() + this->x_ * other.Y() - this->y_ * other.X() + this->z_ * other.W();
    return Quaternion<T>(w, x, y, z);
  }

	/**
	 * Invert the Quaternion, creating a Quaternion with the exact inverse rotation.
	 * @return: A new quaternion holding the roation in the other direction.
	 */
  Quaternion<T> invert( void ) {
    return Quaternion<T>( this->w_, this->x_ * -1, this->y_ * -1, this->z_ * -1);
  }

  /**
   * Basically the getters.
   */
  T W() const { return w_; }
  T X() const { return x_; }
  T Y() const { return y_; }
  T Z() const { return z_; }

  /**
   * Calculate the distance between two Quaternions via:
   * theta = 2 * arccos(|<q1, q2>|)
   * Huynh, D.Q., J Math Imaging Vis, 2009, 35, 155. https://doi.org/10.1007/s10851-009-0161-2
   * Huggins, D.J., J Comput Chem, 2014, 35, 377– 385. https://doi.org/10.1002/jcc.23504
   * @argument other: The other quaternion.
   * @return: The difference in the rotation described by the
   *          quaternions, as an angle.
   */
  T distance(Quaternion<T> other) {
    return 2.0 * acos(fabs(this->w_ * other.W() + 
                      this->x_ * other.X() +
                      this->y_ * other.Y() +
                      this->z_ * other.Z()));
  }

  
  /**
   * Rotates a given vector by the Quaternion.
   * @argument vector: The vector to be rotated.
   * @return: The transformed vector.
   */
  Vec3 rotate(Vec3 vector) {

    Quaternion<T> vecQuat(0, vector[0], vector[1], vector[2]);
    vecQuat = *this * vecQuat;
    vecQuat *= this->invert();
    return Vec3(vecQuat.X(), vecQuat.Y(), vecQuat.Z());
  }

  bool operator==(Quaternion const other) const {
    T w = fabs(this->w_ - other.W());
    T x = fabs(this->x_ - other.X());
    T y = fabs(this->y_ - other.Y());
    T z = fabs(this->z_ - other.Z());
    #ifdef TESTS
    return (w < SMALL && x < SMALL && y < SMALL && z < SMALL);
    #else
    return (w < Constants::SMALL && x < Constants::SMALL && y < Constants::SMALL && z < Constants::SMALL);
    #endif
  }

  bool initialized() const noexcept
  {
    return !(w_ == 0.0 && x_ == 0.0 && y_ == 0.0 && z_ == 0.0);
  }

private:
  T w_;
  T x_;
  T y_;
  T z_;
};
#endif
