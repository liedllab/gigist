/**
 * A new implementation of the GIST calculation. Also useable on the GPU.
 * 
 * @author Johannes Kraml
 * @email Johannes.Kraml@uibk.ac.at
 */

#ifndef ACTION_GIGIST_H
#define ACTION_GIGIST_H

/*
 * Index:
 * Includes.......................................25
 *    General includes............................25
 *    Cpptraj specific includes...................33
 *    Cuda specific includes......................47
 *    OpenMP specific includes....................52
 * Definitions....................................57
 * Classes........................................79
 * 		DataDictionary..............................79
 * 		Action_GIGist...............................176
 */


#include <vector>
#include <math.h>
#include <exception>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <map>
#include <memory>

#include "Action.h"
#include "Vec3.h"
#include "ImagedAction.h"
#include "CpptrajStdio.h"
#include "Constants.h"
#include "DataSet_3D.h"
#include "ProgressBar.h"
#include "DataSet_GridFlt.h"
#include "DataFile.h"
#include "Timer.h"
#include "Quaternion.h"
#include "ExceptionsGIST.h"


#ifdef CUDA
#include "cuda_kernel_gist/GistCudaSetup.cuh"
#endif


#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef HUGE
#define HUGE 3.40282347e+38F
#endif

#ifdef CUDA
#define DOUBLE_O_FLOAT float
#else
#define DOUBLE_O_FLOAT double
#endif


// Boltzmann Constant
#define BOLTZMANN_CONSTANT 1.38064852E-23
// DEBYE
#define DEBYE 0.20822678





/**
 * Data Dictionary helper class.
 * The different atoms of the solvent can be added to the
 * dictionary via the add command.
 */
class DataDictionary {
private:
  std::vector<std::string> names;
public:
  /**
   * Constructor creates the initial data points, which will always be the same
   * for each GIST run.
   */
  DataDictionary() {
    this->names.push_back("population");
    this->names.push_back("dTStrans_norm");
    this->names.push_back("dTStrans_dens");
    this->names.push_back("dTSorient_norm");
    this->names.push_back("dTSorient_dens");
    this->names.push_back("dTSsix_norm");
    this->names.push_back("dTSsix_dens");
    this->names.push_back("Eww");
    this->names.push_back("Eww_norm");
    this->names.push_back("Eww_dens");
    this->names.push_back("Esw");
    this->names.push_back("Esw_norm");
    this->names.push_back("Esw_dens");
    this->names.push_back("dipole_x");
    this->names.push_back("dipole_y");
    this->names.push_back("dipole_z");
    this->names.push_back("dipole_xtemp");
    this->names.push_back("dipole_ytemp");
    this->names.push_back("dipole_ztemp");
    this->names.push_back("dipole_g");
    this->names.push_back("order");
    this->names.push_back("order_norm");
    this->names.push_back("neighbour");
    this->names.push_back("neighbour_dens");
    this->names.push_back("neighbour_norm");
  }

  /**
   * Calculate the number of data sets.
   * @return: The size of the dictionary.
   */
  unsigned int size( void ) {
    return this->names.size();
  }

  /**
   * Returns the index of a given name, -1, if it is not present.
   * @argument testString: The string to be checked.
   * @return: The index of the testString, or -1 if testString is
   *           not in the data sets.
   */
  int getIndex(std::string testString) {
    for (unsigned int i = 0; i < this->names.size(); ++i) {
      if (testString.compare(this->names.at(i)) == 0) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Checks whether the dictionary already contains a specific data set.
   * @argument testString: The string to be checked.
   * @return: True if a data set with this name is already there, false otherwise.
   */
  bool contains(std::string testString) {
    return (this->getIndex(testString) != -1);
  }

  /**
   * Get the name of the data set at a given index.
   * @argument idx: The index at which the data set is present.
   * @return: The name of the data set.
   */
  std::string getElement(int idx) {
    return this->names.at(idx);
  }

  /**
   * Add a data set name to the dictionary.
   * @argument name: The name of the new data set.
   */
  void add(std::string name) {
    this->names.push_back(name);
  }

};


/**
 * The Gist class (working on the GPU), implementation is based on the following Papers:
 * 
 * Furthermore, the implementation is also based (in part) on the already present GIST
 * code distributed within cpptraj.
 * Written by Johannes Kraml
 * Johannes.Kraml@uibk.ac.at
 */

class Action_GIGist : public Action {
public:
  // Constructor
  // In: Action_GIGIST.cpp
  // line: 7
  Action_GIGist();
  // Allocator for the object
  // In: Action_GIGIST.h
  // line: 184
  DispatchObject* Alloc() const { return (DispatchObject*) new Action_GIGist(); }
  // Prints the Help message
  // In: Action_GIGIST.cpp
  // line: 35
  void Help() const;
	// Destructor
	// In Action_GIGIST.cpp
	// line: 61
  ~Action_GIGist();
private:
  // Inherited Functions

  // Is called as an initializer of the object
  // In: Action_GIGIST.cpp
  // line: 85
  Action::RetType Init(ArgList&, ActionInit&, int);

  // Is called to setup the calculation with anything topology
  // specific
  // In: Action_GIGIST.cpp
  // line: 211
  Action::RetType Setup(ActionSetup&);

  // Is used to actually perform the action on a single frame
  // In: Action_GIGIST.cpp
  // line: 308
  Action::RetType DoAction(int, ActionFrame&);

  // Is used for postprocessing calculations and output
  // In: Action_GIGIST.cpp
  // line: 671
  void Print();



  // Functions defined to make the programmers life easier

  // In: Action_GIGIST.cpp
  // line: 874
  double calcEnergy(double, int, int);

  // In: Action_GIGIST.cpp
  // line: 886
  double calcDistanceSqrd(ActionFrame&, int, int);

  // In: Action_GIGIST.cpp
  // line: 917
  double calcElectrostaticEnergy(double, int, int);

  // In: Action_GIGIST.cpp
  // line: 935
  double calcVdWEnergy(double, int, int);

  // In: Action_GIGIST.cpp
  // line: 950
  double calcOrientEntropy(int);

  // In: Action_GIGIST.cpp
  // line: 981
  std::vector<double> calcTransEntropy(int);

  // In: Action_GIGIST.cpp
  // line: 1064
  void calcTransEntropyDist(int, int, int, double &, double &);

  // In: Action_GIGIST.cpp
  // line: 1082
  int weight(std::string);

  // In: Action_GIGIST.cpp
  // line: 1105
  void writeDxFile(std::string, std::vector<double>);


  // Functions defined for FEBISS implementation

  // In: Action_GIGIST.cpp
  // line: 1183
  void placeFebissWaters(void);

  // In: Action_GIGIST.cpp
  // line: 1281
  void writeOutSolute(ActionFrame&);

  // In: Action_GIGIST.cpp
  // line: 1311
  void determineGridShells(void);

  // In: Action_GIGIST.cpp
  // line: 1351
  Vec3 coordsFromIndex(const int);

  // In: Action_GIGIST.cpp
  // line: 1367
  void setGridToZero(std::vector<std::vector<std::vector<int>>>&, const int) const;

  // In: Action_GIGIST.cpp
  // line: 1386
  std::tuple<int, int, int, int> findHMaximum(std::vector<std::vector<std::vector<int>>>&, const int, std::tuple<int, int, int, int> = std::make_tuple(0, 0, 0, 0)) const;

  // In: Action_GIGIST.cpp
  // line: 1430
  double calcAngleBetweenHGridPos(const std::tuple<int, int, int, int>&, const std::tuple<int, int, int, int>&) const;

  // In: Action_GIGIST.cpp
  // line: 1449
  void deleteAroundFirstH(std::vector<std::vector<std::vector<int>>>&, const int, const std::tuple<int, int, int, int>&) const;

  // In: Action_GIGIST.cpp
  // line: 1472
  double calcDistSqBetweenHGridPos(const std::tuple<int, int, int, int>&, const std::tuple<int, int, int, int>&) const;

  // In: Action_GIGIST.cpp
  // line: 1490
  Vec3 coordsFromHGridPos(const std::tuple<int, int, int, int>&) const;

  // In: Action_GIGIST.cpp
  // line: 1510
  double assignDensityWeightedDeltaG(const int, const int, const double, const double, const std::vector<double>&, const std::vector<double>&);

  // In: Action_GIGIST.cpp
  // line: 1552
  double addWaterShell(double&, const std::vector<double>&, const int, const int);

  // In: Action_GIGIST.cpp
  // line: 1573
  void subtractWater(std::vector<double> &, const int, const int, const double, const double);

  // In: Action_GIGIST.cpp
  // line: 1610
  void writeFebissPdb(const int, const Vec3&, const Vec3&, const Vec3&, const double);


  // Necessary Variables

  // For CUDA use, store some more elements
#ifdef CUDA
  std::vector<float> lJParamsA_;
  std::vector<float> lJParamsB_;
  std::vector<int> NBIndex_;
  int numberAtomTypes_;

  // Arrays on GPU
  int *NBindex_c_;
  void *molecule_c_;
  void *paramsLJ_c_;
  float *max_c_;
  float *min_c_;
  float *result_w_c_;
  float *result_s_c_;
  int *result_O_c_;
  int *result_N_c_;

  // CUDA only functions

  // In: Action_GIGIST.cpp
  // line: 1087
  void freeGPUMemory(void);

  // In: Action_GIGIST.cpp
  // line: 1112
  void copyToGPU(void);

#endif

  // Dataset pointer
  DataSetList *list_;

  // Necessary information for the computation
  double rho0_;
  unsigned int numberSolvent_;
  std::string centerSolventAtom_;
  unsigned int numberAtoms_;
  double temperature_;
  int nFrames_;
  int forceStart_;
  int centerSolventIdx_;
  int headAtomType_;
  double neighbourCut2_;

  // Topology Object
  Topology *top_;

  // Defining the Grid
  double voxelVolume_;
  double voxelSize_;
  unsigned int nVoxels_;
  Vec3 dimensions_;
  Vec3 center_;
  Vec3 gridStart_;
  Vec3 gridEnd_;
  ImagedAction image_;

  // Vector to store the result
  std::vector<DataSet_3D*> result_;
  std::vector<std::vector<double> > resultV_;

  std::vector<std::vector<Quaternion<DOUBLE_O_FLOAT> > > quaternions_;
  std::vector<DOUBLE_O_FLOAT> charges_;
  std::vector<int> molecule_;
  std::vector<int> atomTypes_;

  // Is a usual array, as std::vector<bool> is actually not a vector storing boolean
  // values but a bit string with the boolean values encoded at each position.
  bool *solvent_;
  // FEBISS related variables
  bool placeWaterMolecules_ = false;
  int nSoluteAtoms_ = 0;
  double idealWaterAngle_ = 104.57;

  std::vector<std::vector<Vec3> > waterCoordinates_;
  std::vector<std::vector<Vec3>> hVectors_;
  std::map<double, std::vector<int>> shellcontainer_;
  std::vector<double> shellcontainerKeys_;
  DataDictionary dict_;

  CpptrajFile *datafile_;
  CpptrajFile *dxfile_;
  CpptrajFile *febissWaterfile_;

  std::vector<int> solventAtomCounter_;
  bool writeDx_;
  bool doorder_;

  Timer tRot_;
  Timer tEadd_;
  Timer tDipole_;
  Timer tHead_;
  Timer tEnergy_;




};

#endif
