#ifndef QUATERNIONG_CUH
#define QUATERNIONG_CUH

#include "../Quaternion.h"
/**
 * The Quaternion class to compare rotation angles.
 */
 template <class T>
 class QuaternionG {
 public:
  __host__
  __device__
  QuaternionG<T>() {
   this->w_ = 0;
   this->x_ = 0;
   this->y_ = 0;
   this->z_ = 0;
  }
   /**
    * Constructor from a given set of quaternion coordinates.
    * @argument w: From the Quaternion theory.
    * @argument x: From the Quaternion theory.
    * @argument y: From the Quaternion theory.
    * @argument z: From the Quaternion theory.
    */
   __host__
   __device__
    QuaternionG<T>(T w, T x, T y, T z) {
     this->w_ = w;
     this->x_ = x;
     this->y_ = y;
     this->z_ = z;
   }
   
   /**
    * Constructor for a given set of angles (yaw, pitch and roll).
    * Please if using this, actually use the Tait-Bryan angles.
    * @argument psi: The psi Tait-Bryan angle (yaw).
    * @argument theta: The theta Tait-Bryan angle ()
    */
    __host__
    __device__
   QuaternionG<T>(T psi, T theta, T phi) {
     T cos_psi   = cos(psi   * 0.5);
     T cos_theta = cos(theta * 0.5);
     T cos_phi   = cos(phi   * 0.5);
 
     T sin_psi   = sin(psi   * 0.5);
     T sin_theta = sin(theta * 0.5);
     T sin_phi   = sin(phi   * 0.5);
 
     this->w_ = cos_psi * cos_theta * cos_phi + sin_psi * sin_theta * sin_phi;
     this->x_ = cos_psi * cos_theta * sin_phi - sin_psi * sin_theta * cos_phi;
     this->y_ = cos_psi * sin_theta * cos_phi + sin_psi * cos_theta * sin_phi;
     this->z_ = sin_psi * cos_theta * cos_phi - cos_psi * sin_theta * sin_phi;
   }
   
   __host__
   QuaternionG<T>(Quaternion<T> other) {
     this->w_ = other.W();
     this->x_ = other.X();
     this->y_ = other.Y();
     this->z_ = other.Z();
   }
 
   /**
    * Basically the getters.
    */
   __host__
   __device__
   T W() { return w_; }
   __host__
   __device__
   T X() { return x_; }
   __host__
   __device__
   T Y() { return y_; }
   __host__
   __device__
   T Z() { return z_; }
 
   /**
    * Calculate the distance between two Quaternions via:
    * theta = 2 * arccos(|<q1, q2>|)
    * @argument other: The other quaternion.
    * @return: The difference in the rotation described by the
    *          quaternions, as an angle.
    */
   __host__
   __device__
   T distance(QuaternionG<T> other) {
     return 2.0 * acos(fabs(this->w_ * other.W() + 
                       this->x_ * other.X() +
                       this->y_ * other.Y() +
                       this->z_ * other.Z()));
   }
   
   __host__
   QuaternionG<T> &operator=(Quaternion<T> other) {
     this->w_ = other.W();
     this->x_ = other.X();
     this->y_ = other.Y();
     this->z_ = other.Z();
     return *this;
   }
 
 private:
   T w_;
   T x_;
   T y_;
   T z_;
 };
 #endif