#include "Action_GIGIST.h"
#include <iostream>
#include <iomanip>

/**
 * Standard constructor
 */
Action_GIGist::Action_GIGist() :
#ifdef CUDA
NBindex_c_(nullptr),
molecule_c_(nullptr),
paramsLJ_c_(nullptr),
max_c_(nullptr), // TODO: unused
min_c_(nullptr), // TODO: unused
result_w_c_(nullptr),
result_s_c_(nullptr),
result_O_c_(nullptr),
result_N_c_(nullptr),
#endif
list_(nullptr),
top_(nullptr),
solvent_(nullptr),
dict_(DataDictionary()),
datafile_(nullptr),
dxfile_(nullptr),
wrongNumberOfAtoms_(false)
{}

/**
 * The help function.
 */
void Action_GIGist::Help() const {
  mprintf("     Usage:\n"
          "    griddim [dimx dimy dimz]   Defines the dimension of the grid.\n"
          "    <gridcntr [x y z]>         Defines the center of the grid, default [0 0 0].\n"
          "    <temp 300>                 Defines the temperature of the simulation.\n"
          "    <gridspacn 0.5>            Defines the grid spacing\n"
          "    <refdens 0.0329>           Defines the reference density for the water model.\n"
          "    <febiss 104.57>            Activates FEBISS placement with given ideal water angle (only available for water)\n"
          "    <out \"out.dat\">          Defines the name of the output file.\n"
          "    <dx>                       Set to write out dx files. Population is always written.\n"
          "    <solventStart n>           Sets the first solvent as the nth molecule.\n"

          "  The griddimensions must be set in integer values and have to be larger than 0.\n"
          "  The greatest advantage, stems from the fact that this code is parallelized\n"
          "  on the GPU.\n\n"

          "  The code is meant to run on the GPU. Therefore, the CPU implementation of GIST\n"
          "  in this code is probably slower than the original GIST implementation.\n\n"

          "  When using this GIST implementation please cite:\n"
          "#    Johannes Kraml, Anna S. Kamenik, Franz Waibl, Michael Schauperl, Klaus R. Liedl, JCTC (2019)\n"
          "#    Steven Ramsey, Crystal Nguyen, Romelia Salomon-Ferrer, Ross C. Walker, Michael K. Gilson, and Tom Kurtzman\n"
          "#      J. Comp. Chem. 37 (21) 2016\n"
          "#    Crystal Nguyen, Michael K. Gilson, and Tom Young, arXiv:1108.4876v1 (2011)\n"
          "#    Crystal N. Nguyen, Tom Kurtzman Young, and Michael K. Gilson,\n"
          "#      J. Chem. Phys. 137, 044101 (2012)\n"
          "#    Lazaridis, J. Phys. Chem. B 102, 3531–3541 (1998)\n");
}

Action_GIGist::~Action_GIGist() {
  // The GPU memory should already be freed, but just in case...
  #ifdef CUDA
  freeGPUMemory();
  #endif
}

void Action_GIGist::calcGridStart() noexcept
{
  info_.grid.start.SetVec(info_.grid.center[0] - (info_.grid.dimensions[0] * 0.5) * info_.grid.voxelSize, 
                          info_.grid.center[1] - (info_.grid.dimensions[1] * 0.5) * info_.grid.voxelSize,
                          info_.grid.center[2] - (info_.grid.dimensions[2] * 0.5) * info_.grid.voxelSize);
}

void Action_GIGist::calcGridEnd() noexcept
{
  info_.grid.end.SetVec(info_.grid.center[0] + info_.grid.dimensions[0] * info_.grid.voxelSize, 
                        info_.grid.center[1] + info_.grid.dimensions[1] * info_.grid.voxelSize,
                        info_.grid.center[2] + info_.grid.dimensions[2] * info_.grid.voxelSize);
}

/***
 * Document in header FILE!
 * 
 */
bool Action_GIGist::analyzeInfo(ArgList &argList)
{
  // System Information
  info_.system.temperature = argList.getKeyDouble("temp", 300.0);
  info_.system.rho0 = argList.getKeyDouble("refdens", 0.0329);
  info_.system.nFrames = 0;

  // GIST related information
  info_.gist.solventStart = argList.getKeyInt("solventStart", -1);
  info_.gist.neighborCutoff = argList.getKeyDouble("neighbour", 3.5);
  info_.gist.neighborCutoff *= info_.gist.neighborCutoff;
  info_.gist.calcEnergy = !(argList.hasKey("skipE"));
  info_.gist.writeDx = argList.hasKey("dx");
  info_.gist.doorder = argList.hasKey("doorder");
  info_.gist.useCOM = argList.hasKey("com");

  // Grid related information
  info_.grid.voxelSize = argList.getKeyDouble("gridspacn", 0.5);
  info_.grid.voxelVolume = info_.grid.voxelSize * info_.grid.voxelSize * info_.grid.voxelSize;

  if (argList.Contains("griddim")) {
    ArgList dimArgs = argList.GetNstringKey("griddim", 3);
    double x = dimArgs.getNextInteger(-1.0);
    double y = dimArgs.getNextInteger(-1.0);
    double z = dimArgs.getNextInteger(-1.0);
    if ( (x <= 0) || (y <= 0) || (z <= 0) ) {
      mprinterr("Error: griddimension must be positive integers (non zero).\n\n");
      return false;
    }
    info_.grid.dimensions.SetVec(x, y, z);
    info_.grid.nVoxels = info_.grid.dimensions[0] * info_.grid.dimensions[1] * info_.grid.dimensions[2];
  } else {
    mprinterr("Error: Dimensions must be set!\n\n");
    return false;
  }

  double x = 0, y = 0, z = 0;
  if (argList.Contains("gridcntr")) {
    ArgList cntrArgs = argList.GetNstringKey("gridcntr", 3);
    x = cntrArgs.getNextDouble(-1);
    y = cntrArgs.getNextDouble(-1);
    z = cntrArgs.getNextDouble(-1);
  } else {
    mprintf("Warning: No grid center specified, defaulting to origin!\n\n");
  }
  info_.grid.center.SetVec(x, y, z);

  if (argList.hasKey("febiss")) {
    placeWaterMolecules_ = true;
  }

  calcGridStart();
  calcGridEnd();

  return true;
}

void Action_GIGist::printCitationInfo() const noexcept
{
  mprintf("Center: %g %g %g, Dimensions %d %d %d\n"
    "  When using this GIST implementation please cite:\n"
    "#    Johannes Kraml, Anna S. Kamenik, Franz Waibl, Michael Schauperl, Klaus R. Liedl, JCTC (2019)\n"
    "#    Steven Ramsey, Crystal Nguyen, Romelia Salomon-Ferrer, Ross C. Walker, Michael K. Gilson, and Tom Kurtzman\n"
    "#      J. Comp. Chem. 37 (21) 2016\n"
    "#    Crystal Nguyen, Michael K. Gilson, and Tom Young, arXiv:1108.4876v1 (2011)\n"
    "#    Crystal N. Nguyen, Tom Kurtzman Young, and Michael K. Gilson,\n"
    "#      J. Chem. Phys. 137, 044101 (2012)\n"
    "#    Lazaridis, J. Phys. Chem. B 102, 3531–3541 (1998)\n",
    info_.grid.center[0],
    info_.grid.center[1],
    info_.grid.center[2],
    static_cast<int>( info_.grid.dimensions[0] ),
    static_cast<int>( info_.grid.dimensions[1] ),
    static_cast<int>( info_.grid.dimensions[2] ) 
  );
}



/**
 * Initialize the GIST calculation by setting up the users input.
 * @param argList: The argument list of the user.
 * @param actionInit: The action initialization object.
 * @return: Action::OK on success and Action::ERR on error.
 */
Action::RetType Action_GIGist::Init(ArgList &argList, ActionInit &actionInit, int test) {
#if defined MPI
  if (actionInit.TrajComm().Size() > 1) {
    mprinterr("Error: GIST cannot yet be used with MPI parallelization.\n"
              "       Maximum allowed processes is 1, you used %d.\n",
              actionInit.TrajComm().Size());
    return Action::ERR;
  }
#endif
  
  // Get Infos
  if (! analyzeInfo(argList) )
    return Action::ERR;
  
  // Imaging
  image_.InitImaging( true );


  std::string outfilename{argList.GetStringKey("out", "out.dat")};
  datafile_ = actionInit.DFL().AddCpptrajFile( outfilename, "GIST output" );

  if (placeWaterMolecules_) {
    hVectors_.resize(info_.grid.nVoxels);
  }

  quaternions_.resize(info_.grid.nVoxels);
  waterCoordinates_.resize(info_.grid.nVoxels);

  std::string dsname{ actionInit.DSL().GenerateDefaultName("GIST") };
  result_ = std::vector<DataSet_3D *>(dict_.size());
  for (unsigned int i = 0; i < dict_.size(); ++i) {
    result_.at(i) = (DataSet_3D*)actionInit.DSL().AddSet(DataSet::GRID_FLT, MetaData(dsname, dict_.getElement(i)));
    result_.at(i)->Allocate_N_C_D(
      info_.grid.dimensions[0],
      info_.grid.dimensions[1],
      info_.grid.dimensions[2],
      info_.grid.center,
      info_.grid.voxelSize
    );

    if (
        ( info_.gist.writeDx &&
          dict_.getElement(i).compare("Eww") != 0 && 
          dict_.getElement(i).compare("Esw") != 0 &&
          dict_.getElement(i).compare("dipole_xtemp") != 0 && 
          dict_.getElement(i).compare("dipole_ytemp") != 0 &&
          dict_.getElement(i).compare("dipole_ztemp") != 0 &&
          dict_.getElement(i).compare("order") != 0 && 
          dict_.getElement(i).compare("neighbour") != 0 ) ||
        i == 0 
       ) 
    {
      DataFile *file = actionInit.DFL().AddDataFile(dict_.getElement(i) + ".dx");
      file->AddDataSet(result_.at(i));
    }
  }
  if (placeWaterMolecules_)
    this->febissWaterfile_ = actionInit.DFL().AddCpptrajFile( "febiss-waters.pdb", "GIST output");

  printCitationInfo();
  
  return Action::OK;
}

/**
 * Setup for the GIST calculation. Does everything involving the Topology file.
 * @param setup: The setup object of the cpptraj code libraries.
 * @return: Action::OK on success, Action::ERR otherwise.
 */
Action::RetType Action_GIGist::Setup(ActionSetup &setup) {
  solventAtomCounter_ = std::vector<int>();
  // Setup imaging and topology parsing.
  image_.SetupImaging( setup.CoordInfo().TrajBox().Type() );

  // Save topology and topology related values
  top_             = setup.TopAddress();
  info_.system.numberAtoms = setup.Top().Natom();
  info_.system.numberSolvent   = setup.Top().Nsolvent();
  solvent_ = std::make_unique<bool []>(info_.system.numberAtoms);
  bool firstRound{ true };

  // Save different values, which depend on the molecules and/or atoms.
  for (Topology::mol_iterator mol = setup.Top().MolStart(); 
          mol != setup.Top().MolEnd(); ++mol) {
    int nAtoms{ static_cast<int>(mol->NumAtoms()) };

    for (int i = 0; i < nAtoms; ++i) {
      molecule_.push_back( setup.Top()[mol->BeginAtom() + i].MolNum() );
      charges_.push_back( setup.Top()[mol->BeginAtom() + i].Charge() );
      atomTypes_.push_back( setup.Top()[mol->BeginAtom() + i].TypeIndex() );
      masses_.push_back( setup.Top()[mol->BeginAtom() + i].Mass());

      // Check if the molecule is a solvent, either by the topology parameters or because force was set.
      if ( (mol->IsSolvent() && info_.gist.solventStart == -1) 
              || (( info_.gist.solventStart > -1 ) 
                          && 
                  ( setup.Top()[mol->BeginAtom()].MolNum() >= info_.gist.solventStart ))
        ) {
        std::string aName{ setup.Top()[mol->BeginAtom() + i].ElementName() };
        
        // Check if dictionary already holds an entry for the atoms name, if not add it to
        // the dictionary, if yes, add 1 to the correct solvent atom counter.
        if (! (dict_.contains(aName)) ) {
          dict_.add(aName);
          solventAtomCounter_.push_back(1);
        } else if (firstRound) {
          solventAtomCounter_.at(dict_.getIndex(aName) - result_.size()) += 1;
        }

        // Check for the centerSolventAtom (which in this easy approximation is either C or O)
        if ( weight(aName) < weight(info_.gist.centerAtom) ) {
          info_.gist.centerAtom = setup.Top()[mol->BeginAtom() + i].ElementName();
          info_.gist.centerIdx = i; // Assumes the same order of atoms.
          info_.gist.centerType = setup.Top()[mol->BeginAtom() + i].TypeIndex();
        }
        // Set solvent to true
        solvent_[mol->BeginAtom() + i] = true;
      } else {
        solvent_[mol->BeginAtom() + i] = false;
      }
    }
    if ((mol->IsSolvent() && info_.gist.solventStart == -1) || 
          (
            ( info_.gist.solventStart > -1 ) 
            && 
            ( setup.Top()[mol->BeginAtom()].MolNum() >= info_.gist.solventStart )
          )
    ) {
      firstRound = false;
    }
  }
  // Add results for the different solvent atoms.
  for (unsigned int i = 0; i < (dict_.size() - result_.size()); ++i) {
    resultV_.push_back(
      std::vector<double>(
        info_.grid.dimensions[0] *
        info_.grid.dimensions[1] *
        info_.grid.dimensions[2]
      )
    );
  }

  
  // Define different things for the case that this was compiled using CUDA
#ifdef CUDA
  NonbondParmType nb{ setup.Top().Nonbond() };
  NBIndex_ = nb.NBindex();
  numberAtomTypes_ = nb.Ntypes();
  for (unsigned int i = 0; i < nb.NBarray().size(); ++i) {
    lJParamsA_.push_back( (float) nb.NBarray().at(i).A() );
    lJParamsB_.push_back( (float) nb.NBarray().at(i).B() );
  }

  try {
    allocateCuda(((void**)&NBindex_c_), NBIndex_.size() * sizeof(int));
    allocateCuda((void**)&max_c_, 3 * sizeof(float));
    allocateCuda((void**)&min_c_, 3 * sizeof(float));
    allocateCuda((void**)&result_w_c_, numberAtoms_ * sizeof(float));
    allocateCuda((void**)&result_s_c_, numberAtoms_ * sizeof(float));
    allocateCuda((void**)&result_O_c_, numberAtoms_ * 4 * sizeof(int));
    allocateCuda((void**)&result_N_c_, numberAtoms_ * sizeof(int));
  } catch (CudaException &e) {
    mprinterr("Error: Could not allocate memory on GPU!\n");
    freeGPUMemory();
    return Action::ERR;
  }
  try {
    copyToGPU();
  } catch (CudaException &e) {
    return Action::ERR;
  }
#endif
  return Action::OK;
}

/**
 * Starts the calculation of GIST. Can use either CUDA, OPENMP or single thread code.
 * This function is actually way too long. Refactoring of this code might help with
 * readability.
 * @param frameNum: The number of the frame.
 * @param frame: The frame itself.
 * @return: Action::ERR on error, Action::OK if everything ran smoothly.
 */
Action::RetType Action_GIGist::DoAction(int frameNum, ActionFrame &frame) {

  info_.system.nFrames++;
  std::vector<DOUBLE_O_FLOAT> eww_result(info_.system.numberAtoms);
  std::vector<DOUBLE_O_FLOAT> esw_result(info_.system.numberAtoms);
  std::vector<std::vector<int> > order_indices{};
  

  if (placeWaterMolecules_ && info_.system.nFrames == 1) {
    this->writeOutSolute(frame);
  }


  if (info_.gist.useCOM && info_.system.nFrames == 1) 
  {
    for (Topology::mol_iterator mol = top_->MolStart(); mol < top_->MolEnd(); ++mol) {
      int moleculeLength = mol->EndAtom() - mol->BeginAtom() + 1;
      if (moleculeLength < 3)
         continue;
      if ((mol->IsSolvent() && info_.gist.solventStart == -1 ) || 
        (
          ( info_.gist.solventStart > -1 ) &&
          ( top_->operator[](mol->BeginAtom()).MolNum() >= info_.gist.solventStart )
          )
      ) {
	      quat_indices_ = calcQuaternionIndices(mol->BeginAtom(), mol->EndAtom(), frame.Frm().XYZ(mol->BeginAtom()));
        break;
      }
    }
  }
  
  // CUDA necessary information
  #ifdef CUDA
  std::vector<int> result_o{ std::vector<int>(4 * info_.system.numberAtoms) };
  std::vector<int> result_n{ std::vector<int>(info_.system.numberAtoms) };
  if (info_.gist.calcEnergy){
  tEnergy_.Start();

  Matrix_3x3 ucell_m{}, recip_m{};
  float *recip = nullptr;
  float *ucell = nullptr;
  int boxinfo{};

  // Check Boxinfo and write the necessary data into recip, ucell and boxinfo.
  switch(image_.ImageType()) {
    case NONORTHO:
      recip = new float[9];
      ucell = new float[9];
      frame.Frm().BoxCrd().ToRecip(ucell_m, recip_m);
      for (int i = 0; i < 9; ++i) {
        ucell[i] = static_cast<float>( ucell_m.Dptr()[i] );
        recip[i] = static_cast<float>( recip_m.Dptr()[i] );
      }
      boxinfo = 2;
      break;
    case ORTHO:
      recip = new float[9];
      for (int i = 0; i < 3; ++i) {
        recip[i] = static_cast<float>( frame.Frm().BoxCrd()[i] );
      }
      ucell = nullptr;
      boxinfo = 1;
      break;
    case NOIMAGE:
      recip = nullptr;
      ucell = nullptr;
      boxinfo = 0;
      break;
    default:
      mprinterr("Error: Unexpected box information found.");
      return Action::ERR;
  }

  // std::vector<int> result_o{ std::vector<int>(4 * info_.system.numberAtoms) };
  // std::vector<int> result_n{ std::vector<int>(info_.system.numberAtoms) };
  // TODO: Switch things around a bit and move the back copying to the end of the calculation.
  //       Then the time needed to go over all waters and the calculations that come with that can
  //			 be hidden quite nicely behind the interaction energy calculation.
  // Must create arrays from the vectors, does that by getting the address of the first element of the vector.
  std::vector<std::vector<float> > e_result{ 
    doActionCudaEnergy(
      frame.Frm().xAddress(),
      NBindex_c_,
      numberAtomTypes_,
      paramsLJ_c_,
      molecule_c_,
      boxinfo,
      recip,
      ucell,
      info_.syste.numberAtoms,
      min_c_,
      max_c_,
      info_.system.headAtomType,
      info_.gist.neigborCutoff,
      &(result_o[0]),
      &(result_n[0]),
      result_w_c_,
      result_s_c_,
      result_O_c_,
      result_N_c_,
      info_.gist.doorder) };
  eww_result = e_result.at(0);
  esw_result = e_result.at(1);

  if (info_.gist.doorder) {
    int counter{ 0 };
    for (unsigned int i = 0; i < (4 * info_.system.numberAtoms); i += 4) {
      ++counter;
      std::vector<int> temp{};
      for (unsigned int j = 0; j < 4; ++j) {
        temp.push_back(result_o.at(i + j));
      }
      order_indices.push_back(temp);
    }
  }

  delete[] recip;
  delete[] ucell;

  tEnergy_.Stop();
  }
  #endif

  std::vector<bool> onGrid(top_->Natom());
  for (unsigned int i = 0; i < onGrid.size(); ++i) {
    onGrid.at(i) = false;
  }

  #if defined _OPENMP && defined CUDA
  tHead_.Start();
  #pragma omp parallel for
  #endif
  for (Topology::mol_iterator mol = top_->MolStart(); mol < top_->MolEnd(); ++mol) {
    if ((mol->IsSolvent() && info_.gist.solventStart == -1) || 
      (
        ( info_.gist.solventStart > -1 ) &&
        ( top_->operator[](mol->BeginAtom()).MolNum() >= info_.gist.solventStart )
      )
    ) {  
      int headAtomIndex{ -1 };
      // Keep voxel at -1 if it is not possible to put it on the grid
      int voxel{ -1 };
      std::vector<Vec3> molAtomCoords{};
      Vec3 com{ 0, 0, 0 };

      // If center of mass should be used, use this part.
      if (info_.gist.useCOM) {
        int mol_begin{ mol->BeginAtom() };
        int mol_end{ mol->EndAtom() };
        com = calcCenterOfMass(mol_begin, mol_end, frame.Frm().XYZ(mol_begin)) ;
        voxel = bin(mol_begin, mol_end, com, frame);
      }
      

      #if !defined _OPENMP && !defined CUDA
      tHead_.Start();
      #endif
      for (int atom1 = mol->BeginAtom(); atom1 < mol->EndAtom(); ++atom1) {
        bool first{ true };
        if (solvent_[atom1]) { // Do we need that?
          // Save coords for later use.
          const double *vec = frame.Frm().XYZ(atom1);
          molAtomCoords.push_back(Vec3(vec));
          // Check if atom is "Head" atom of the solvent
          // Could probably save some time here by writing head atom indices into an array.
          // TODO: When assuming fixed atom position in topology, should be very easy.
          if ( !info_.gist.useCOM && std::string((*top_)[atom1].ElementName()).compare(info_.gist.centerAtom) == 0 && first ) {
            // Try to bin atom1 onto the grid. If it is possible, get the index and keep working,
            // if not, calculate the energies between all atoms to this point.
            voxel = bin(mol->BeginAtom(), mol->EndAtom(), vec, frame);
            headAtomIndex = atom1 - mol->BeginAtom();
            first = false;
          } else {
            size_t bin_i{}, bin_j{}, bin_k{};
            if ( result_.at(dict_.getIndex("population"))->Bin().Calc(vec[0], vec[1], vec[2], bin_i, bin_j, bin_k) 
                    /*&& bin_i < dimensions_[0] && bin_j < dimensions_[1] && bin_k < dimensions_[2]*/) {
              std::string aName{ top_->operator[](atom1).ElementName() };
              long voxTemp{ result_.at(dict_.getIndex("population"))->CalcIndex(bin_i, bin_j, bin_k) };
              #ifdef _OPENMP
              #pragma omp critical
              {
              #endif
              try{
                resultV_.at(dict_.getIndex(aName) - result_.size()).at(voxTemp) += 1.0;
              } catch(std::out_of_range e)
              {
                std::cout << std::setprecision(30) << (size_t)((vec[0] + 35.0f) / 0.5f) << ", " << vec[1] << ", " << vec[2] << '\n';
                std::cout << result_.at(dict_.getIndex("population"))->Bin().Calc(vec[0], vec[1], vec[2], bin_i, bin_j, bin_k) << '\n';
                std::cout << bin_i << " " << bin_j << " " << bin_k << '\n';
                std::cout << voxTemp << '\n';
                throw std::out_of_range("");
              }
              #ifdef _OPENMP
              }
              #endif
            }
          }
        }
      }
      #if !defined _OPENMP && !defined CUDA
      tHead_.Stop();
      #endif

      if (voxel != -1) {
        this->result_.at(this->dict_.getIndex("population"))->UpdateVoxel(voxel, 1.0);
        
        #if !defined _OPENMP && !defined CUDA
        this->tRot_.Start();
        #endif
        Vec3 X;
        Vec3 Y;
        bool setX = false;
        bool setY = false;

        for (unsigned int i = 0; i < molAtomCoords.size(); ++i) {
          if ((int)i != headAtomIndex) {
            if (setX && !setY) {
              Y.SetVec(molAtomCoords.at(i)[0] - molAtomCoords.at(headAtomIndex)[0], 
                        molAtomCoords.at(i)[1] - molAtomCoords.at(headAtomIndex)[1], 
                        molAtomCoords.at(i)[2] - molAtomCoords.at(headAtomIndex)[2]);
              if (placeWaterMolecules_)
                this->hVectors_.at(voxel).push_back(Vec3(Y[0], Y[1], Y[2]));
              Y.Normalize();
              setY = true;
            }
            if (!setX) {
              X.SetVec(molAtomCoords.at(i)[0] - molAtomCoords.at(headAtomIndex)[0], 
                        molAtomCoords.at(i)[1] - molAtomCoords.at(headAtomIndex)[1], 
                        molAtomCoords.at(i)[2] - molAtomCoords.at(headAtomIndex)[2]);
              if (placeWaterMolecules_)
                this->hVectors_.at(voxel).push_back(Vec3(X[0], X[1], X[2]));
              X.Normalize();
              setX = true;
            }
            if (setX && setY) {
              break;
            }
          }
        }

        Quaternion<DOUBLE_O_FLOAT> quat{};
        
        // Create Quaternion for the rotation from the new coordintate system to the lab coordinate system.
        if (!info_.gist.useCOM) {
          quat = calcQuaternion(molAtomCoords, molAtomCoords.at(headAtomIndex), headAtomIndex);
        } else {
          // -1 Will never evaluate to true, so in the funciton it will have no consequence.
          quat = calcQuaternion(molAtomCoords, com, quat_indices_);
        }
        //if (quat.initialized())
        //{
          #ifdef _OPENMP
          #pragma omp critical
          {
          #endif
          quaternions_.at(voxel).push_back(quat);
          #ifdef _OPENMP
          }
          #endif
        //}
        

        #if !defined _OPENMP && !defined CUDA
        tRot_.Stop();
        #endif
        
  // If energies are already here, calculate the energies right away.
  #ifdef CUDA
        /* 
        * Calculation of the order parameters
        * Following formula:
        * q = 1 - 3/8 * SUM[a>b]( cos(Thet[a,b]) + 1/3 )**2
        * This, however, only makes sense for water, so please do not
        * use it for any other solvent.
        */
        if (info_.gist.doorder) {
          double sum{ 0 };
          Vec3 cent{ frame.Frm().xAddress() + (mol->BeginAtom() + headAtomIndex) * 3 };
          std::vector<Vec3> vectors{};
          switch(image_.ImageType()) {
            case NONORTHO:
            case ORTHO:
              {
                Matrix_3x3 ucell, recip;
                frame.Frm().BoxCrd().ToRecip(ucell, recip);
                Vec3 vec(frame.Frm().xAddress() + (order_indices.at(mol->BeginAtom() + headAtomIndex).at(0) * 3));
                vectors.push_back( MinImagedVec(vec, cent, ucell, recip));
                vec = Vec3(frame.Frm().xAddress() + (order_indices.at(mol->BeginAtom() + headAtomIndex).at(1) * 3));
                vectors.push_back( MinImagedVec(vec, cent, ucell, recip));
                vec = Vec3(frame.Frm().xAddress() + (order_indices.at(mol->BeginAtom() + headAtomIndex).at(2) * 3));
                vectors.push_back( MinImagedVec(vec, cent, ucell, recip));
                vec = Vec3(frame.Frm().xAddress() + (order_indices.at(mol->BeginAtom() + headAtomIndex).at(3) * 3));
                vectors.push_back( MinImagedVec(vec, cent, ucell, recip));
              }
              break;
            default:
              vectors.push_back( Vec3( frame.Frm().xAddress() + (order_indices.at(mol->BeginAtom() + headAtomIndex).at(0) * 3) ) - cent );
              vectors.push_back( Vec3( frame.Frm().xAddress() + (order_indices.at(mol->BeginAtom() + headAtomIndex).at(1) * 3) ) - cent );
              vectors.push_back( Vec3( frame.Frm().xAddress() + (order_indices.at(mol->BeginAtom() + headAtomIndex).at(2) * 3) ) - cent );
              vectors.push_back( Vec3( frame.Frm().xAddress() + (order_indices.at(mol->BeginAtom() + headAtomIndex).at(3) * 3) ) - cent );
          }
          
          for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 4; ++j) {
              double cosThet{ (vectors.at(i) * vectors.at(j)) / sqrt(vectors.at(i).Magnitude2() * vectors.at(j).Magnitude2()) };
              sum += (cosThet + 1.0/3) * (cosThet + 1.0/3);
            }
          }
          #ifdef _OPENMP
          #pragma omp critical
          {
          #endif
          result_.at(dict_.getIndex("order"))->UpdateVoxel(voxel, 1.0 - (3.0/8.0) * sum);
          #ifdef _OPENMP
          }
          #endif
        }
        #ifdef _OPENMP
        #pragma omp critical
        {
        #endif
        result_.at(dict_.getIndex("neighbour"))->UpdateVoxel(voxel, result_n.at(mol->BeginAtom() + headAtomIndex));
        #ifdef _OPENMP
        }
        #endif
        // End of calculation of the order parameters

        #ifndef _OPENMP
        tEadd_.Start();
        #endif
        #ifdef _OPENMP
        #pragma omp critical
        {
        #endif
        // There is absolutely nothing to check here, as the solute can not be in place here.
        for (int atom = mol->BeginAtom(); atom < mol->EndAtom(); ++atom) {
          // Just adds up all the interaction energies for this voxel.
          result_.at(dict_.getIndex("Eww"))->UpdateVoxel(voxel, static_cast<double>(eww_result.at(atom)));
          result_.at(dict_.getIndex("Esw"))->UpdateVoxel(voxel, static_cast<double>(esw_result.at(atom)));
        }
        #ifdef _OPENMP
        }
        #endif
        #ifndef _OPENMP
        tEadd_.Stop();
        #endif
  #endif
      }

      // If CUDA is used, energy calculations are already done.
  #ifndef CUDA
      if (voxel != -1 ) {
        std::vector<Vec3> nearestWaters(4);
        // Use HUGE distances at the beginning. This is defined as 3.40282347e+38F.
        double distances[4]{HUGE, HUGE, HUGE, HUGE};
        // Needs to be fixed, one does not need to calculate all interactions each time.
        for (int atom1 = mol->BeginAtom(); atom1 < mol->EndAtom(); ++atom1) {
          double eww{ 0 };
          double esw{ 0 };
  // OPENMP only over the inner loop

          #pragma omp parallel for
          for (unsigned int atom2 = 0; atom2 < info_.system.numberAtoms; ++atom2) {
            if ( (*top_)[atom1].MolNum() != (*top_)[atom2].MolNum() ) {
              tEadd_.Start();
              double r_2{ calcDistanceSqrd(frame, atom1, atom2) };
              double energy{ calcEnergy(r_2, atom1, atom2) };
              tEadd_.Stop();
              if (solvent_[atom2]) {
                #pragma omp atomic
                eww += energy;
              } else {
                #pragma omp atomic
                esw += energy;
              }
              if (atomTypes_.at(atom1) == info_.gist.centerType &&
                  atomTypes_.at(atom2) == info_.gist.centerType) {
                if (r_2 < distances[0]) {
                  distances[3] = distances[2];
                  distances[2] = distances[1];
                  distances[1] = distances[0];
                  distances[0] = r_2;
                  nearestWaters.at(3) = nearestWaters.at(2);
                  nearestWaters.at(2) = nearestWaters.at(1);
                  nearestWaters.at(1) = nearestWaters.at(0);
                  nearestWaters.at(0) = Vec3(frame.Frm().XYZ(atom2)) - Vec3(frame.Frm().XYZ(atom1));
                } else if (r_2 < distances[1]) {
                  distances[3] = distances[2];
                  distances[2] = distances[1];
                  distances[1] = r_2;
                  nearestWaters.at(3) = nearestWaters.at(2);
                  nearestWaters.at(2) = nearestWaters.at(1);
                  nearestWaters.at(1) = Vec3(frame.Frm().XYZ(atom2)) - Vec3(frame.Frm().XYZ(atom1));
                } else if (r_2 < distances[2]) {
                  distances[3] = distances[2];
                  distances[2] = r_2;
                  nearestWaters.at(3) = nearestWaters.at(2);
                  nearestWaters.at(2) = Vec3(frame.Frm().XYZ(atom2)) - Vec3(frame.Frm().XYZ(atom1));
                } else if (r_2 < distances[3]) {
                  distances[3] = r_2;
                  nearestWaters.at(3) = Vec3(frame.Frm().XYZ(atom2)) - Vec3(frame.Frm().XYZ(atom1));
                }
                if (r_2 < info_.gist.neighborCutoff) {
                  #ifdef _OPENMP
                  #pragma omp critical
                  {
                  #endif
                  result_.at(dict_.getIndex("neighbour"))->UpdateVoxel(voxel, 1);
                  #ifdef _OPENMP
                  }
                  #endif
                }
              }
            }
          }
          double sum{ 0 };
          for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 4; ++j) {
              double cosThet{ (nearestWaters.at(i) * nearestWaters.at(j)) / 
                                sqrt(nearestWaters.at(i).Magnitude2() * nearestWaters.at(j).Magnitude2()) };
              sum += (cosThet + 1.0/3) * (cosThet + 1.0/3);
            }
          }
          #ifdef _OPENMP
          #pragma omp critical
          {
          #endif
          result_.at(dict_.getIndex("order"))->UpdateVoxel(voxel, 1.0 - (3.0/8.0) * sum);
          eww /= 2.0;
          result_.at(dict_.getIndex("Eww"))->UpdateVoxel(voxel, eww);
          result_.at(dict_.getIndex("Esw"))->UpdateVoxel(voxel, esw);
          #ifdef _OPENMP
          }
          #endif
        }
      }
#endif
    }
  }
  
  #if defined _OPENMP && defined CUDA
  tHead_.Stop();
  #endif

  return Action::OK;
}

/**
 * Post Processing is done here.
 */
void Action_GIGist::Print() {
  /* This is not called for two reasons
   * 1) The RAM on the GPU is far less than the main memory
   * 2) It does not speed up the calculation significantly enough
   * However, this can be changed if wished for (It is not yet stable enough to be used)
   * Tests are ongoing
   */
  #ifdef CUDA_UPDATED
  std::vector<std::vector<float> > dTSTest = doActionCudaEntropy(waterCoordinates_, info_.grid,dimension[0], info_.grid,dimension[1],
                                                              info_.grid,dimension[2], quaternions_, info_.system.temperature, info_.system.rho0, info_.system.nFrames);
  #endif
  mprintf("Processed %d frames.\nMoving on to entropy calculation.\n", info_.system.nFrames);
  ProgressBar progBarEntropy(info_.grid.nVoxels);
#ifdef _OPENMP
  int curVox{ 0 };
#endif
  #pragma omp parallel for
  for (unsigned int voxel = 0; voxel < info_.grid.nVoxels; ++voxel) {
    // If _OPENMP is defined, the progress bar has to be updated critically,
    // to ensure the right addition.
#ifndef _OPENMP
    progBarEntropy.Update( voxel );
#else
    #pragma omp critical
    progBarEntropy.Update( curVox++ );
#endif
    double dTSorient_norm   { 0.0 };
    double dTStrans_norm    { 0.0 };
    double dTSsix_norm      { 0.0 };
    double dTSorient_dens   { 0.0 };
    double dTStrans_dens    { 0.0 };
    double dTSsix_dens      { 0.0 };
    double Esw_norm         { 0.0 };
    double Esw_dens         { 0.0 };
    double Eww_norm         { 0.0 };
    double Eww_dens         { 0.0 };
    double order_norm       { 0.0 };
    double neighbour_dens   { 0.0 };
    double neighbour_norm   { 0.0 };
    // Only calculate if there is actually water molecules at that position.
    if (result_.at(dict_.getIndex("population"))->operator[](voxel) > 0) {
      double pop = result_.at(dict_.getIndex("population"))->operator[](voxel);
      // Used for calcualtion of the Entropy on the GPU
      #ifdef CUDA_UPDATED
      dTSorient_norm          = dTSTest.at(1).at(voxel);
      dTStrans_norm           = dTSTest.at(0).at(voxel);
      dTSsix_norm             = dTSTest.at(2).at(voxel);
      dTSorient_dens          = dTSorient_norm * pop / (this->nFrames_ * this->voxelVolume_);
      dTStrans_dens           = dTStrans_norm * pop / (this->nFrames_ * this->voxelVolume_);
      dTSsix_dens             = dTSsix_norm * pop / (this->nFrames_ * this->voxelVolume_);
      #else
      std::vector<double> dTSorient = this->calcOrientEntropy(voxel);
      dTSorient_norm          = dTSorient.at(0);
      dTSorient_dens          = dTSorient.at(1);
      std::vector<double> dTS = this->calcTransEntropy(voxel);
      dTStrans_norm           = dTS.at(0);
      dTStrans_dens           = dTS.at(1);
      dTSsix_norm             = dTS.at(2);
      dTSsix_dens             = dTS.at(3);
      #endif
      
      Esw_norm = result_.at(dict_.getIndex("Esw"))->operator[](voxel) / pop;
      Esw_dens = result_.at(dict_.getIndex("Esw"))->operator[](voxel) / (info_.system.nFrames * info_.grid.voxelVolume);
      Eww_norm = result_.at(dict_.getIndex("Eww"))->operator[](voxel) / pop;
      Eww_dens = result_.at(dict_.getIndex("Eww"))->operator[](voxel) / (info_.system.nFrames * info_.grid.voxelVolume);
      order_norm = result_.at(dict_.getIndex("order"))->operator[](voxel) / pop;
      neighbour_norm = result_.at(dict_.getIndex("neighbour"))->operator[](voxel) / pop;
      neighbour_dens = result_.at(dict_.getIndex("neighbour"))->operator[](voxel) / (info_.system.nFrames * info_.grid.voxelVolume);
    }
    
    // Calculate the final dipole values. The temporary data grid has to be used, as data
    // already saved cannot be updated.
    double DPX{ result_.at(dict_.getIndex("dipole_xtemp"))->operator[](voxel) / (DEBYE * info_.system.nFrames * info_.grid.voxelVolume) };
    double DPY{ result_.at(dict_.getIndex("dipole_ytemp"))->operator[](voxel) / (DEBYE * info_.system.nFrames * info_.grid.voxelVolume) };
    double DPZ{ result_.at(dict_.getIndex("dipole_ztemp"))->operator[](voxel) / (DEBYE * info_.system.nFrames * info_.grid.voxelVolume) };
    double DPG{ sqrt( DPX * DPX + DPY * DPY + DPZ * DPZ ) };
    result_.at(dict_.getIndex("dTStrans_norm"))->UpdateVoxel(voxel, dTStrans_norm);
    result_.at(dict_.getIndex("dTStrans_dens"))->UpdateVoxel(voxel, dTStrans_dens);
    result_.at(dict_.getIndex("dTSorient_norm"))->UpdateVoxel(voxel, dTSorient_norm);
    result_.at(dict_.getIndex("dTSorient_dens"))->UpdateVoxel(voxel, dTSorient_dens);
    result_.at(dict_.getIndex("dTSsix_norm"))->UpdateVoxel(voxel, dTSsix_norm);
    result_.at(dict_.getIndex("dTSsix_dens"))->UpdateVoxel(voxel, dTSsix_dens);
    result_.at(dict_.getIndex("order_norm"))->UpdateVoxel(voxel, order_norm);
    result_.at(dict_.getIndex("neighbour_norm"))->UpdateVoxel(voxel, neighbour_norm);
    result_.at(dict_.getIndex("neighbour_dens"))->UpdateVoxel(voxel, neighbour_dens);

    
    result_.at(dict_.getIndex("Esw_norm"))->UpdateVoxel(voxel, Esw_norm);
    result_.at(dict_.getIndex("Esw_dens"))->UpdateVoxel(voxel, Esw_dens);
    result_.at(dict_.getIndex("Eww_norm"))->UpdateVoxel(voxel, Eww_norm);
    result_.at(dict_.getIndex("Eww_dens"))->UpdateVoxel(voxel, Eww_dens);
    // Maybe there is a better way, I have to look that
    result_.at(dict_.getIndex("dipole_x"))->UpdateVoxel(voxel, DPX);
    result_.at(dict_.getIndex("dipole_y"))->UpdateVoxel(voxel, DPY);
    result_.at(dict_.getIndex("dipole_z"))->UpdateVoxel(voxel, DPZ);
    result_.at(dict_.getIndex("dipole_g"))->UpdateVoxel(voxel, DPG);
    for (unsigned int i = 0; i < resultV_.size(); ++i) {
      resultV_.at(i).at(voxel) /= (info_.system.nFrames * info_.grid.voxelVolume * info_.system.rho0 * solventAtomCounter_.at(i));
    }
  }

  if (placeWaterMolecules_) {
    if (info_.gist.centerAtom == "O" && solventAtomCounter_.size() == 2) {
      placeFebissWaters();
    } else {
      mprinterr("Error: FEBISS only works with water as solvent so far.\n");
    }
  }

  
  mprintf("Writing output:\n");
  this->datafile_->Printf("GIST calculation output. rho0 = %g, n_frames = %d\n", this->rho0_, this->nFrames_);
  this->datafile_->Printf("   voxel        x          y          z         population     dTSt_d(kcal/mol)  dTSt_n(kcal/mol)"
                          "  dTSo_d(kcal/mol)  dTSo_n(kcal/mol)  dTSs_d(kcal/mol)  dTSs_n(kcal/mol)   "
                          "Esw_d(kcal/mol)   Esw_n(kcal/mol)   Eww_d(kcal/mol)   Eww_n(kcal/mol)    dipoleX    "
                          "dipoleY    dipoleZ    dipole    neighbour_d    neighbour_n    order_n  ");
  // Moved the densities to the back of the output file, so that the energies are always
  // at the same positions.
  for (unsigned int i = result_.size(); i < dict_.size(); ++i) {
    datafile_->Printf("  g_%s  ", dict_.getElement(i).c_str());
  }
  datafile_->Printf("\n");

  // Final output, the DX files are done automatically by cpptraj
  // so only the standard GIST-format is done here
  ProgressBar progBarIO(info_.grid.nVoxels);
  for (unsigned int voxel = 0; voxel < info_.grid.nVoxels; ++voxel) {
    progBarIO.Update( voxel );
    size_t i{}, j{}, k{};
    result_.at(dict_.getIndex("population"))->ReverseIndex(voxel, i, j, k);
    Vec3 coords{ result_.at(dict_.getIndex("population"))->Bin().Center(i, j, k) };
    datafile_->Printf("%d %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g", 
                            voxel, coords[0], coords[1], coords[2],
                            result_.at(dict_.getIndex("population"))->operator[](voxel),
                            result_.at(dict_.getIndex("dTStrans_dens"))->operator[](voxel),
                            result_.at(dict_.getIndex("dTStrans_norm"))->operator[](voxel),
                            result_.at(dict_.getIndex("dTSorient_dens"))->operator[](voxel),
                            result_.at(dict_.getIndex("dTSorient_norm"))->operator[](voxel),
                            result_.at(dict_.getIndex("dTSsix_dens"))->operator[](voxel),
                            result_.at(dict_.getIndex("dTSsix_norm"))->operator[](voxel),
                            result_.at(dict_.getIndex("Esw_dens"))->operator[](voxel),
                            result_.at(dict_.getIndex("Esw_norm"))->operator[](voxel),
                            result_.at(dict_.getIndex("Eww_dens"))->operator[](voxel),
                            result_.at(dict_.getIndex("Eww_norm"))->operator[](voxel),
                            result_.at(dict_.getIndex("dipole_x"))->operator[](voxel),
                            result_.at(dict_.getIndex("dipole_y"))->operator[](voxel),
                            result_.at(dict_.getIndex("dipole_z"))->operator[](voxel),
                            result_.at(dict_.getIndex("dipole_g"))->operator[](voxel),
                            result_.at(dict_.getIndex("neighbour_dens"))->operator[](voxel),
                            result_.at(dict_.getIndex("neighbour_norm"))->operator[](voxel),
                            result_.at(dict_.getIndex("order_norm"))->operator[](voxel)
                            );
    for (unsigned int i = 0; i < resultV_.size(); ++i) {
      datafile_->Printf(" %g", resultV_.at(i).at(voxel));
    }
    datafile_->Printf("\n");
  }
  // The atom densities of the solvent compared to the reference density.
  if (info_.gist.writeDx) {
    for (int i = 0; i < static_cast<int>( resultV_.size() ); ++i) {
      writeDxFile("g_" + dict_.getElement(result_.size() + i) + ".dx", resultV_.at(i));
    }
  }


  mprintf("Timings:\n"
          " Find Head Atom:   %8.3f\n"
          " Add up Energy:    %8.3f\n"
          " Calculate Dipole: %8.3f\n"
          " Calculate Quat:   %8.3f\n"
          " Calculate Energy: %8.3f\n\n",
          tHead_.Total(),
          tEadd_.Total(),
          tDipole_.Total(),
          tRot_.Total(),
          tEnergy_.Total());
  if (wrongNumberOfAtoms_)
  {
    mprintf("Warning: It seems you are having multiple solvents in your system.");
  }
  #ifdef CUDA
  freeGPUMemory();
  #endif
}

/**
 * Calculate the Van der Waals and electrostatic energy.
 * @param r_2: The squared distance between atom 1 and atom 2.
 * @param a1: The first atom.
 * @param a2: The second atom.
 * @return: The interaction energy between the two atoms.
 */
double Action_GIGist::calcEnergy(double r_2, int a1, int a2) {
  r_2 = 1 / r_2;
  return calcElectrostaticEnergy(r_2, a1, a2) + calcVdWEnergy(r_2, a1, a2);
}

/**
 * Calculate the squared distance between two atoms.
 * @param frm: The frame for which to calculate the distance.
 * @param a1: The first atom for the calculation.
 * @param a2: The second atom for the calculation.
 * @return: The squared distance between the two atoms.
 */
double Action_GIGist::calcDistanceSqrd(const ActionFrame &frm, int a1, int a2) {
    Matrix_3x3 ucell{}, recip{};
    double dist{ 0.0 };
    Vec3 vec1{frm.Frm().XYZ(a1)};
    Vec3 vec2{frm.Frm().XYZ(a2)};
    switch( image_.ImageType() ) {
        case NONORTHO:
            frm.Frm().BoxCrd().ToRecip(ucell, recip);
            dist = DIST2_ImageNonOrtho(vec1, vec2, ucell, recip);
            break;
        case ORTHO:
            dist = DIST2_ImageOrtho(vec1, vec2, frm.Frm().BoxCrd());
            break;
        case NOIMAGE:
            dist = DIST2_NoImage(vec1, vec2);
            break;
        default:
            throw BoxInfoException();
    }
    return dist;
}

/**
 * Calculate the electrostatic energy between two atoms, as
 * follows from:
 * E(el) = q1 * q2 / r
 * @param r_2_i: The inverse of the squared distance between the atoms.
 * @param a1: The atom index of atom 1.
 * @param a2: The atom index of atom 2.
 * @return: The electrostatic energy.
 */
double Action_GIGist::calcElectrostaticEnergy(double r_2_i, int a1, int a2) {
    //double q1 = top_->operator[](a1).Charge();
    //double q2 = top_->operator[](a2).Charge();
    double q1{ charges_.at(a1) };
    double q2{ charges_.at(a2) };
    return q1 * Constants::ELECTOAMBER * q2 * Constants::ELECTOAMBER * sqrt(r_2_i);
}

/**
 * Calculate the van der Waals interaction energy between
 * two different atoms, as follows:
 * E(vdw) = A / (r ** 12) - B / (r ** 6)
 * Be aware that the inverse is used, as to calculate faster.
 * @param r_2_i: The inverse of the squared distance between the two atoms.
 * @param a1: The atom index of atom1.
 * @param a2: The atom index of atom2.
 * @return: The VdW interaction energy.
 */
double Action_GIGist::calcVdWEnergy(double r_2_i, int a1, int a2) {
    // Attention, both r_6 and r_12 are actually inverted. This is very ok, and makes the calculation faster.
    // However, it is not noted, thus it could be missleading
    double r_6{ r_2_i * r_2_i * r_2_i };
    double r_12{ r_6 * r_6 };
    NonbondType const &params = top_->GetLJparam(a1, a2);
    return params.A() * r_12 - params.B() * r_6;
}

/**
 * Calculate the orientational entropy of the water atoms
 * in a given voxel.
 * @param voxel: The index of the voxel.
 * @return: The entropy of the water molecules in that voxel.
 */
std::vector<double> Action_GIGist::calcOrientEntropy(int voxel) {
  std::vector<double> ret(2);
  int nwtotal = this->result_.at(this->dict_.getIndex("population"))->operator[](voxel);
  if(nwtotal < 2) {
    return ret;
  }
  double dTSo_n{ 0.0 };
  int water_count{ 0 };
  for (int n0 = 0; n0 < nwtotal; ++n0) {
    double NNr{ 100000.0 };
    for (int n1 = 0; n1 < nwtotal; ++n1) {
      if (n1 == n0) {
        continue;
      }

     if ( quaternions_.at(voxel).at(n0).initialized() &&
		      quaternions_.at(voxel).at(n1).initialized() )
     {
      	double rR{ quaternions_.at(voxel).at(n0).distance(quaternions_.at(voxel).at(n1)) };
      	if ( (rR < NNr) && (rR > 0.0) ) {
        	NNr = rR;
      	}
      }
    }
    if (NNr < 99999 && NNr > 0) {
      ++water_count;
      dTSo_n += log(NNr * NNr * NNr / (3.0 * Constants::TWOPI));
    }
  }
  dTSo_n += water_count * log(water_count);
  dTSo_n = Constants::GASK_KCAL * this->temperature_ * (dTSo_n / water_count + Constants::EULER_MASC);
  ret.at(0) = dTSo_n;
  ret.at(1) = dTSo_n * water_count / (this->nFrames_ * this->voxelVolume_);
  return ret;
}

/**
 * Calculate the translational entropy.
 * @param voxel: The voxel for which to calculate the translational entropy.
 * @return: A vector type object, holding the values for the translational
 *          entropy, as well as the six integral entropy.
 */
std::vector<double> Action_GIGist::calcTransEntropy(int voxel) {
  // Will hold dTStrans (norm, dens) and dTSsix (norm, dens)
  std::vector<double> ret(4);
  // dTStrans uses all solvents => use nwtotal
  // dTSsix does not use ions => count separately
  int nwtotal = (*this->result_.at(this->dict_.getIndex("population")))[voxel];
  int nw_six{ 0 };
  for (int n0 = 0; n0 < nwtotal; ++n0) {
    double NNd = HUGE;
    double NNs = HUGE;
    if ( this->quaternions_[voxel][n0].initialized() ) {
        // the current molecule has rotational degrees of freedom, i.e., it's not an ion.
        ++nw_six;
    }
    // Self is not migrated to the function, as it would need to have a check against self comparison
    // The need is not entirely true, as it would produce 0 as a value.
    for (int n1 = 0; n1 < nwtotal; ++n1) {
      if (n1 == n0) {
        continue;
      }
      double dd{ (waterCoordinates_.at(voxel).at(n0) - waterCoordinates_.at(voxel).at(n1)).Magnitude2() };
      if (dd > Constants::SMALL && dd < NNd) {
        NNd = dd;
      }
      if ( quaternions_.at(voxel).at(n0).initialized() &&
		      quaternions_.at(voxel).at(n1).initialized() )
      {
      	double rR{ quaternions_.at(voxel).at(n0).distance(quaternions_.at(voxel).at(n1)) };
      	double ds{ rR * rR + dd };
      	if (ds < NNs && ds > Constants::SMALL) {
        	NNs = ds;
      	}
      }
    }
    // Get the values for the dimensions
    Vec3 step{};
    step.SetVec(info_.grid.dimensions[2] * info_.grid.dimensions[1], info_.grid.dimensions[2], 1);
    int x{ voxel / (static_cast<int>( info_.grid.dimensions[2] ) * static_cast<int>( info_.grid.dimensions[1]) ) };
    int y{ (voxel / static_cast<int>( info_.grid.dimensions[2] ) ) % static_cast<int>( info_.grid.dimensions[1] ) };
    int z{ voxel % static_cast<int>( info_.grid.dimensions[2] ) };
    // Check if the value is inside the boundaries, i.e., there is still a neighbour in
    // each direction.
    if ( !((x <= 0                       ) || (y <= 0                       ) || (z <= 0                       ) ||
           (x >= info_.grid.dimensions[0] - 1) || (y >= info_.grid.dimensions[1] - 1) || (z >= info_.grid.dimensions[2] - 1)) ) {
      // Builds a 3 x 3 x 3 cube for entropy calculation

      for (int dim1 = -1; dim1 < 2; ++dim1) {
        for (int dim2 = -1; dim2 < 2; ++dim2) {
          for (int dim3 = -1; dim3 < 2; ++dim3) {
            int voxel2{ voxel + dim1 * static_cast<int>( step[0] ) + dim2 * static_cast<int>( step[1] ) + dim3 * static_cast<int>( step[2] ) };
            // Checks if the voxel2 is the same as voxel, if so, has already been calculated
            if (voxel2 != voxel) {
              calcTransEntropyDist(voxel, voxel2, n0, NNd, NNs);
            }
          }
        }
      }
      // Bring NNd back to power 1. NNs is kept at power 2, since it is needed as a power 3 later.
      NNd = sqrt(NNd);
    } else {
      ret.at(0) = 0;
      ret.at(1) = 0;
      ret.at(2) = 0;
      ret.at(3) = 0;
      return ret;
    }
    if (NNd < 3 && NNd > 0) {
      // For both, the number of frames is used as the number of measurements.
      // The third power of NNd has to be taken, since NNd is only power 1.
      ret.at(0) += log(NNd * NNd * NNd * info_.system.nFrames * 4 * Constants::PI * info_.system.rho0 / 3.0);
      // NNs is used to the power of 6, since it is already power of 2, only the third power
      // has to be calculated.
      if ( this->quaternions_[voxel][n0].initialized() ) {
        ret.at(2) += log(NNs * NNs * NNs * this->nFrames_ * Constants::PI * this->rho0_ / 48.0);
      }
    }
  }
  if (ret.at(0) != 0) {
    double dTSt_n{ Constants::GASK_KCAL * this->temperature_ * (ret.at(0) / nwtotal + Constants::EULER_MASC) };
    ret.at(0) = dTSt_n;
    ret.at(1) = dTSt_n * nwtotal / (this->nFrames_ * this->voxelVolume_);
    double dTSs_n{ Constants::GASK_KCAL * this->temperature_ * (ret.at(2) / nw_six + Constants::EULER_MASC) };
    ret.at(2) = dTSs_n;
    ret.at(3) = dTSs_n * nw_six / (this->nFrames_ * this->voxelVolume_);
  }
  return ret;
}

/**
 * Calculates the distance between the different atoms in the voxels.
 * Both, for the distance in space, as well as the angular distance.
 * @param voxel1: The first of the two voxels.
 * @param voxel2: The voxel to compare voxel1 to.
 * @param n0: The water molecule of the first voxel.
 * @param NNd: The lowest distance in space. If the calculated one
 *                is smaller, saves it here.
 * @param NNs: The lowest distance in angular space. If the calculated
 *                one is smaller, saves it here.
 */
void Action_GIGist::calcTransEntropyDist(int voxel1, int voxel2, int n0, double &NNd, double &NNs) {
  int nSolvV2{ static_cast<int>( (*result_.at(dict_.getIndex("population")))[voxel2] ) };
  for (int n1 = 0; n1 < nSolvV2; ++n1) {
    if (voxel1 == voxel2 && n0 == n1) {
      continue;
    }
    double dd{ (waterCoordinates_.at(voxel1).at(n0) - waterCoordinates_.at(voxel2).at(n1)).Magnitude2() };
    if (dd > Constants::SMALL && dd < NNd) {
      NNd = dd;
    }
    if (quaternions_.at(voxel1).at(n0).initialized() &&
		    quaternions_.at(voxel2).at(n1).initialized())
    {
      double rR{ quaternions_.at(voxel1).at(n0).distance(quaternions_.at(voxel2).at(n1)) };
      double ds{ rR * rR + dd };
      if (ds < NNs && ds > 0) {
        NNs = ds;
      }
    }
  }
} 

/**
 * A weighting for the different elements.
 * @param atom: A string holding the element symbol.
 * @return a weight for that particular element.
 **/
int Action_GIGist::weight(std::string atom) {
  if (atom.compare("S") == 0) {
    return 0;
  }
  if (atom.compare("C") == 0) {
    return 1;
  }
  if (atom.compare("O") == 0) {
    return 2;
  }
  if (atom.compare("") == 0) {
    return 10000;
  }
  return 1000;
}

/**
 * Writes a dx file. The dx file is the same file as the cpptraj dx file, however,
 * this is under my complete control, cpptraj is not.
 * Still for most calculations, the cpptraj tool is used.
 * @param name: A string holding the name of the written file.
 * @param data: The data to write to the dx file.
 */
void Action_GIGist::writeDxFile(std::string name, const std::vector<double> &data) {
  std::ofstream file{};
  file.open(name.c_str());
  Vec3 origin{ info_.grid.center - info_.grid.dimensions * (0.5 * info_.grid.voxelSize) };
  file << "object 1 class gridpositions counts " << info_.grid.dimensions[0] << " " << info_.grid.dimensions[1] << " " << info_.grid.dimensions[2] << "\n";
  file << "origin " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
  file << "delta " << info_.grid.voxelSize << " 0 0\n";
  file << "delta 0 " << info_.grid.voxelSize << " 0\n";
  file << "delta 0 0 " << info_.grid.voxelSize << "\n";
  file << "object 2 class gridconnections counts " << info_.grid.dimensions[0] << " " << info_.grid.dimensions[1] << " " << info_.grid.dimensions[2] << "\n";
  file << "object 3 class array type double rank 0 items " << info_.grid.nVoxels << " data follows" << "\n";
  int i{ 0 };
  while ( (i + 3) < static_cast<int>( info_.grid.nVoxels  )) {
    file << data.at(i) << " " << data.at(i + 1) << " " << data.at(i + 2) << "\n";
    i +=3;
  }
  while (i < static_cast<int>( info_.grid.nVoxels )) {
    file << data.at(i) << " ";
    i++;
  }
  file << std::endl;
}

/**
 * Calculate the center of mass for a set of atoms. These atoms do not necessarily need
 * to belong to the same molecule, but in this case do.
 * @param atom_begin: The first atom in the set.
 * @param atom_end: The index of the last atom in the set.
 * @param coords: The current coordinates, on which processing occurs.
 * @return A vector, of class Vec3, holding the center of mass.
 */
Vec3 Action_GIGist::calcCenterOfMass(int atom_begin, int atom_end, const double *coords) {
  double mass{ 0.0 };
  double x{ 0.0 }, y{ 0.0 }, z{ 0.0 };
  for (int i = 0; i < (atom_end - atom_begin); ++i) {
    double currentMass{masses_.at(i + atom_begin)};
    x += coords[i * 3    ] * currentMass;
    y += coords[i * 3 + 1] * currentMass;
    z += coords[i * 3 + 2] * currentMass;
    mass += currentMass;
  }
  return Vec3{x / mass, y / mass, z / mass};
}

/**
 * A function to bin a certain vector to a grid. This still does more, will be fixed.
 * @param begin: The first atom in the molecule.
 * @param end: The last atom in the molecule.
 * @param vec: The vector to be binned.
 * @param frame: The current frame.
 * @return The voxel this frame was binned into. If binning was not succesfull, returns -1.
 */
int Action_GIGist::bin(int begin, int end, const Vec3 &vec, const ActionFrame &frame) {
  size_t bin_i{}, bin_j{}, bin_k{};
  // This is set to -1, if binning is not possible, the function will return a nonsensical value of -1, which can be tested.
  int voxel{ -1 };
  if (result_.at(dict_.getIndex("population"))->Bin().Calc(vec[0], vec[1], vec[2], bin_i, bin_j, bin_k)
      /*&& bin_i < dimensions_[0] && bin_j < dimensions_[1] && bin_k < dimensions_[2]*/)
  {
    voxel = result_.at(dict_.getIndex("population"))->CalcIndex(bin_i, bin_j, bin_k);
    
    // Does not necessarily need this in this function
    try {
      waterCoordinates_.at(voxel).push_back(vec);
    } catch (std::out_of_range e)
    {
      std::cout << info_.system.nFrames << '\n';
      throw e;
    }

    #ifdef _OPENMP
    #pragma omp critical
    {
    #endif
    
      result_.at(dict_.getIndex("population"))->UpdateVoxel(voxel, 1.0);
    if (!info_.gist.useCOM) {
      resultV_.at(dict_.getIndex(info_.gist.centerAtom) - result_.size()).at(voxel) += 1.0;
    }
    #ifdef _OPENMP
    }
    #endif

    calcDipole(begin, end, voxel, frame);

  }
  return voxel;
}

/**
 * Calculates the total dipole for a given set of atoms.
 * @param begin: The index of the first atom of the set.
 * @param end: The index of the last atom of the set.
 * @param voxel: The voxel in which the values should be binned
 * @param frame: The current frame.
 * @return Nothing at the moment
 */
void Action_GIGist::calcDipole(int begin, int end, int voxel, const ActionFrame &frame) {
  #if !defined _OPENMP && !defined CUDA
    tDipole_.Start();
#endif
    double DPX{ 0 };
    double DPY{ 0 };
    double DPZ{ 0 };
    for (int atoms = begin; atoms < end; ++atoms)
    {
      const double *XYZ = frame.Frm().XYZ(atoms);
      double charge{ charges_.at(atoms) };
      DPX += charge * XYZ[0];
      DPY += charge * XYZ[1];
      DPZ += charge * XYZ[2];
    }

    #ifdef _OPENMP
    #pragma omp critical
    {
    #endif
    result_.at(dict_.getIndex("dipole_xtemp"))->UpdateVoxel(voxel, DPX);
    result_.at(dict_.getIndex("dipole_ytemp"))->UpdateVoxel(voxel, DPY);
    result_.at(dict_.getIndex("dipole_ztemp"))->UpdateVoxel(voxel, DPZ);
    #ifdef _OPENMP
    }
    #endif
#if !defined _OPENMP && !defined CUDA
    tDipole_.Stop();
#endif
}

std::vector<int> Action_GIGist::calcQuaternionIndices(int begin, int end, const double * molAtomCoords)
{
  std::vector<int> indices;
  Vec3 com {calcCenterOfMass( begin, end, molAtomCoords )};
  Vec3 X{ 0, 0, 0};
  for (int i {0}; i < (end - begin); i++)
  {
    if (top_->operator[](i).Element() == Atom::HYDROGEN){
      continue;
    }
    Vec3 coord{&molAtomCoords[i * 3]};
    if ( (coord - com).Length() > 0.2)
    {
      // Return if enough atoms are found
      if (indices.size() >= 2)
      {
        return indices;
      }
      else if (indices.size() == 0)
      {
        indices.push_back(i);
        X = coord - com;
      }
      else
      {
        if ( (X * (coord - com)) / (X.Length() * (coord - com).Length()) <= 0.9)
        {
          indices.push_back(i);
          return indices;
        }
      }
    }
  }
  if (indices.size() < 2) {
    for (int i {0}; i < (end - begin); i++)
    {
      if (top_->operator[](i).Element() == Atom::HYDROGEN){
        Vec3 coord{&molAtomCoords[i * 3]};
        if ( (coord - com).Length() > 0.2)
        {
          // Return if enough atoms are found
          if (indices.size() >= 2)
          {
            return indices;
          }
          else if (indices.size() == 0)
          {
            indices.push_back(i);
            X = coord - com;
          }
          else
          {
            if ( (X * (coord - com)) / (X.Length() * (coord - com).Length()) <= 0.9)
            {
              indices.push_back(i);
              return indices;
            }
          }
        }
      }
    }
  }
  return indices;
}


Quaternion<DOUBLE_O_FLOAT> Action_GIGist::calcQuaternion(const std::vector<Vec3> &molAtomCoords, const Vec3 &center, std::vector<int> indices)
{
  if (static_cast<int>(molAtomCoords.size()) < indices.at(0) || 
          static_cast<int>(molAtomCoords.size()) < indices.at(1))
  {
    wrongNumberOfAtoms_ = true;
    return Quaternion<DOUBLE_O_FLOAT> {};
  }

  Vec3 X = molAtomCoords.at(indices.at(0)) - center;
  Vec3 Y = molAtomCoords.at(indices.at(1)) - center;

  // Create Quaternion for the rotation from the new coordintate system to the lab coordinate system.
   Quaternion<DOUBLE_O_FLOAT> quat(X, Y);
   // The Quaternion would create the rotation of the lab coordinate system onto the
   // calculated solvent coordinate system. The invers quaternion is exactly the rotation of
   // the solvent coordinate system onto the lab coordinate system.
   quat.invert();
   return quat;
}


/**
 * Calculate the quaternion as a rotation when a certain center is given
 * and a set of atomic coordinates are supplied. If the center coordinates
 * are actually one of the atoms, headAtomIndex should evaluate to that
 * atom, if this is not done, unexpexted behaviour might occur.
 * If the center is set to something other than an atomic position,
 * headAtomIndex should evaluate to a nonsensical number (preferrably
 * a negative value).
 * @param molAtomCoords: The set of atomic cooordinates, saved as a vector
 *                           of Vec3 objects.
 * @param center: The center coordinates.
 * @param headAtomIndex: The index of the head atom, when counting the first
 *                           atom as 0, as indices naturally do.
 * @return: A quaternion holding the rotational value.
 * FIXME: Decision for the different X and Y coordinates has to be done at the beginning.
 */
Quaternion<DOUBLE_O_FLOAT> Action_GIGist::calcQuaternion(const std::vector<Vec3> &molAtomCoords, const Vec3 &center, int headAtomIndex) {
  Vec3 X{};
  Vec3 Y{};
  bool setX{false};
  bool setY{false};    
  for (unsigned int i = 0; i < molAtomCoords.size(); ++i) {
    if ((int)i != headAtomIndex) {
      if (setX && !setY) {
        Y.SetVec(molAtomCoords.at(i)[0] - center[0], 
                  molAtomCoords.at(i)[1] - center[1], 
                  molAtomCoords.at(i)[2] - center[2]);
        Y.Normalize();
        
      }
      if (!setX) {
        X.SetVec(molAtomCoords.at(i)[0] - center[0], 
                  molAtomCoords.at(i)[1] - center[1], 
                  molAtomCoords.at(i)[2] - center[2]);
        if (!(X.Length() < 0.001)) {
          X.Normalize();
          setX = true;
        }
      }
      if (setX && setY) {
        break;
      }
    }
  }

  if (X.Length() <= 0.1 || Y.Length() <= 0.1)
  {
    return Quaternion<DOUBLE_O_FLOAT>{};
  }

   // Create Quaternion for the rotation from the new coordintate system to the lab coordinate system.
   Quaternion<DOUBLE_O_FLOAT> quat(X, Y);
   // The Quaternion would create the rotation of the lab coordinate system onto the
   // calculated solvent coordinate system. The invers quaternion is exactly the rotation of
   // the solvent coordinate system onto the lab coordinate system.
   quat.invert();
   return quat;
}

/**
 * Checks for two numbers being almost equal. Also takes care of problems arising due to values close to zero.
 * This comaprison is implemented as suggested by 
 * https://www.learncpp.com/cpp-tutorial/relational-operators-and-floating-point-comparisons/
 * @param input: The input number that should be compared.
 * @param control: The control number to which input should be almost equal.
 * @return true if they are almost equal, false otherwise.
 */
bool Action_GIGist::almostEqual(double input, double control) 
{
  double abs_inp{std::fabs(input)};
  double abs_cont{std::fabs(control)};
  double abs_diff{std::abs(input - control)};

  // Check if the absolute error is already smaller than epsilon (errors close to 0)
  if (abs_diff < __DBL_EPSILON__) {
    return true;
  }

  // Fall back to Knuth's algorithm
  return abs_diff <= ( (abs_inp < abs_cont) ? abs_cont : abs_inp) * __DBL_EPSILON__;
}



// Functions used when CUDA is specified.
#ifdef CUDA

/**
 * Frees all the Memory on the GPU.
 */
void Action_GIGist::freeGPUMemory(void) {
  freeCuda(NBindex_c_);
  freeCuda(molecule_c_);
  freeCuda(paramsLJ_c_);
  freeCuda(max_c_);
  freeCuda(min_c_);
  freeCuda(result_w_c_);
  freeCuda(result_s_c_);
  freeCuda(result_O_c_);
  freeCuda(result_N_c_);
  NBindex_c_   = nullptr;
  molecule_c_  = nullptr;
  paramsLJ_c_  = nullptr;
  max_c_     = nullptr;
  min_c_     = nullptr;
  result_w_c_= nullptr;
  result_s_c_= nullptr;
  result_O_c_  = nullptr;
  result_N_c_  = nullptr;
}

/**
 * Copies data from the CPU to the GPU.
 * @throws: CudaException
 */
void Action_GIGist::copyToGPU(void) {
  float grids[3]{ static_cast<float>(gridStart_[0]), static_cast<float>(gridStart_[1]), static_cast<float>(gridStart_[2]) };
  float gride[3]{ static_cast<float>(gridEnd_[0]), static_cast<float>(gridEnd_[1]), static_cast<float>(gridEnd_[2]) };
  try {
    copyMemoryToDevice(&(NBIndex_[0]), NBindex_c_, NBIndex_.size() * sizeof(int));
    copyMemoryToDevice(grids, min_c_, 3 * sizeof(float));
    copyMemoryToDevice(gride, max_c_, 3 * sizeof(float));
    copyMemoryToDeviceStruct(&(charges_[0]), &(atomTypes_[0]), solvent_, &(molecule_[0]), numberAtoms_, &(molecule_c_),
                              &(lJParamsA_[0]), &(lJParamsB_[0]), lJParamsA_.size(), &(paramsLJ_c_));
  } catch (CudaException &ce) {
    freeGPUMemory();
    mprinterr("Error: Could not copy data to the device.\n");
    throw ce;
  } catch (std::exception &e) {
    freeGPUMemory();
    throw e;
  }
}
#endif

/**
 * @brief main function for FEBISS placement
 *
 */
void Action_GIGist::placeFebissWaters(void) {
  mprintf("Transfering data for FEBISS placement\n");
  determineGridShells();
  /* calculate delta G and read density data */
  std::vector<double> deltaG;
  std::vector<double> relPop;
  for (unsigned int voxel = 0; voxel < info_.grid.nVoxels; ++voxel) {
    double dTSt = result_.at(dict_.getIndex("dTStrans_norm"))->operator[](voxel);
    double dTSo = result_.at(dict_.getIndex("dTSorient_norm"))->operator[](voxel);
    double esw = result_.at(dict_.getIndex("Esw_norm"))->operator[](voxel);
    double eww = result_.at(dict_.getIndex("Eww_norm"))->operator[](voxel);
    double value = esw + eww - dTSo - dTSt;
    deltaG.push_back(value);
    relPop.push_back(resultV_.at(info_.gist.centerIdx).at(voxel));
  }
  /* Place water to recover 95% of the original density */
  int waterToPosition = static_cast<int>(round(info_.system.numberSolvent * 0.95 / 3));
  mprintf("Placing %d FEBISS waters\n", waterToPosition);
  ProgressBar progBarFebiss(waterToPosition);
  /* cycle to position all waters */
  for (int i = 0; i < waterToPosition; ++i) {
    progBarFebiss.Update(i);
    double densityValueOld = 0.0;
    /* get data of current highest density voxel */
    std::vector<double>::iterator maxDensityIterator;
    maxDensityIterator = std::max_element(relPop.begin(), relPop.end());
    double densityValue = *(maxDensityIterator);
    int index = maxDensityIterator - relPop.begin();
    Vec3 voxelCoords = coordsFromIndex(index);
    /**
     * bin hvectors to grid
     * Currently this grid is hardcoded. It has 21 voxels in x, y, z direction.
     * The spacing is 0.1 A, so it stretches out exactly 1 A in each direction
     * and is centered on the placed oxygen. This grid allows for convenient and
     * fast binning of the relative vectors of the H atoms during the simulation
     * which have been stored in hVectors_ as a std::vector of Vec3
     */
    std::vector<std::vector<std::vector<int>>> hGrid;
    int hDim = 21;
    setGridToZero(hGrid, hDim);
    std::vector<int> binnedHContainer;
    /* cycle relative H vectors */
    for (unsigned int j = 0; j < hVectors_[index].size(); ++j) {
      /* cycle x, y, z */
      for (int k = 0; k < 3; ++k) {
        double tmpBin = hVectors_[index][j][k];
        tmpBin *= 10;
        tmpBin = std::round(tmpBin);
        tmpBin += 10;
        /* sometimes bond lengths can be longer than 1 A and exactly along the
         * basis vectors, this ensures that the bin is inside the grid */
        if (tmpBin < 0)
          tmpBin = 0;
        else if (tmpBin > 20)
          tmpBin = 20;
        binnedHContainer.push_back(static_cast<int>(tmpBin));
      }
    }
    /* insert data into grid */
    for (unsigned int j = 0; j < binnedHContainer.size(); j += 3) {
      int x = binnedHContainer[j];
      int y = binnedHContainer[j + 1];
      int z = binnedHContainer[j + 2];
      hGrid[x][y][z] += 1;
    }
    /* determine maxima in grid */
    auto maximum1 = findHMaximum(hGrid, hDim);
    deleteAroundFirstH(hGrid, hDim, maximum1);
    auto maximum2 = findHMaximum(hGrid, hDim, maximum1);
    Vec3 h1 = coordsFromHGridPos(maximum1);
    Vec3 h2 = coordsFromHGridPos(maximum2);
    /* increase included shells until enough density to subtract */
    int shellNum = 0;
    int maxShellNum = shellcontainerKeys_.size() - 1;
    /* not enough density and not reached limit */
    while (densityValue < 1 / (info_.grid.voxelVolume * info_.system.rho0) &&
           shellNum < maxShellNum
    ) {
      densityValueOld = densityValue;
      ++shellNum;
      /* new density by having additional watershell */
      densityValue = addWaterShell(densityValue, relPop, index, shellNum);
    }
    /* determine density weighted delta G with now reached density */
    double weightedDeltaG = assignDensityWeightedDeltaG(
        index, shellNum, densityValue, densityValueOld, relPop, deltaG);
    int atomNumber = 3 * i + nSoluteAtoms_; // running index in pdb file
    /* write new water to pdb */
    writeFebissPdb(atomNumber, voxelCoords, h1, h2, weightedDeltaG);
    /* subtract density in included shells */
    subtractWater(relPop, index, shellNum, densityValue, densityValueOld);
  } // cycle of placed water molecules
}

/**
 * @brief writes solute of given frame into pdb
 *
 * @argument actionFrame frame from which solute is written
 */
void Action_GIGist::writeOutSolute(ActionFrame& frame) {
  std::vector<Vec3> soluteCoords;
  std::vector<std::string> soluteEle;
  for (Topology::mol_iterator mol = top_->MolStart();
       mol < top_->MolEnd(); ++mol) {
    if (!mol->IsSolvent()) {
      for (int atom = mol->BeginAtom(); atom < mol->EndAtom(); ++atom) {
        nSoluteAtoms_++;
        const double *vec = frame.Frm().XYZ(atom);
        soluteCoords.push_back(Vec3(vec));
        soluteEle.push_back(top_->operator[](atom).ElementName());
      }
    }
  }
  for (unsigned int i = 0; i < soluteCoords.size(); ++i) {
    auto name = soluteEle[i].c_str();
    febissWaterfile_->Printf(
      "ATOM  %5d  %3s SOL     1    %8.3f%8.3f%8.3f%6.2f%7.2f          %2s\n",
      i + 1,
      name,
      soluteCoords[i][0],
      soluteCoords[i][1],
      soluteCoords[i][2],
      1.0,
      0.0,
      name
    );
  }
}

/**
 * @brief calculates distance of all voxels to the grid center and groups
 *        them intp shells of identical distances
 *
 * a map stores lists of indices with their identical squared distance as key
 * the indices are stores as difference to the center index to be applicable for
 * all voxels later without knowing the center index
 * a list contains all keys in ascending order to systematically grow included
 * shells later in the algorithm
 */
void Action_GIGist::determineGridShells(void) {
  /* determine center index */
  size_t centeri, centerj, centerk;
  result_.at(dict_.getIndex("population"))->
      Bin().Calc(info_.grid.center[0], info_.grid.center[1], info_.grid.center[2], centeri, centerj, centerk);
  int centerIndex = result_.at(dict_.getIndex("population"))
      ->CalcIndex(centeri, centerj, centerk);
  /* do not use center_ because it does not align with a voxel but lies between voxels */
  /* however the first shell must be solely the voxel itself -> use coords from center voxel */
  Vec3 centerCoords = coordsFromIndex(centerIndex);
  for (unsigned int vox = 0; vox < info_.grid.nVoxels; ++vox) {
    /* determine squared distance */
    Vec3 coords = coordsFromIndex(vox);
    Vec3 difference = coords - centerCoords;
    double distSquared = difference[0] * difference[0] +
                         difference[1] * difference[1] +
                         difference[2] * difference[2];
    /* find function of map returns last memory address of map if not found */
    /* if is entered if distance already present as key in map -> can be added */
    if (shellcontainer_.find(distSquared) != shellcontainer_.end()) {
      shellcontainer_[distSquared].push_back(vox-centerIndex);
    } else {
      /* create new entry in map */
      std::vector<int> indexDifference;
      indexDifference.push_back(vox-centerIndex);
      shellcontainer_.insert(std::make_pair(distSquared, indexDifference));
    }
  }
  /* create list to store ascending keys */
  shellcontainerKeys_.reserve(shellcontainer_.size());
  std::map<double, std::vector<int>>::iterator it = shellcontainer_.begin();
  while(it != shellcontainer_.end()) {
    shellcontainerKeys_.push_back(it->first);
    it++;
  }
}

/**
 * @brief own utility function to get coords from grid index
 *
 * @argument index The index in the GIST grid
 * @return Vec3 coords at the voxel
 */
Vec3 Action_GIGist::coordsFromIndex(const int index) {
  size_t i, j, k;
  result_.at(dict_.getIndex("population"))->ReverseIndex(index, i, j, k);
  Vec3 coords = info_.grid.start;
  /* the + 0.5 * size is necessary because of cpptraj's interprets start as corner of grid */
  /* and voxel coordinates are given for the center of the voxel -> hence the shift of half spacing */
  coords[0] += (i + 0.5) * info_.grid.voxelSize;
  coords[1] += (j + 0.5) * info_.grid.voxelSize;
  coords[2] += (k + 0.5) * info_.grid.voxelSize;
  return coords;
}

/**
 * @brief sets the grid for the H atoms to zero
 *
 * @argument grid
 * @argument dim The dimension of the grid
 */
void Action_GIGist::setGridToZero(std::vector<std::vector<std::vector<int>>>& grid, const int dim) const {
  grid.resize(dim);
  for (int i = 0; i < dim; ++i) {
    grid[i].resize(dim);
    for (int j = 0; j < dim; ++j) {
      grid[i][j].resize(dim);
    }
  }
}

/**
 * @brief Find highest value with possible consideration of first maximum
 *
 * @argument grid
 * @argument dim The dimension of the grid
 * @argument firstMaximum Optional first maximum
 *
 * @return std::tuple<int, int, int, int> maximum in the grid
 */
std::tuple<int, int, int, int> Action_GIGist::findHMaximum(std::vector<std::vector<std::vector<int>>>& grid, const int dim, std::tuple<int, int, int, int> firstMaximum) const {
  std::tuple<int, int, int, int> maximum = std::make_tuple(0, 0, 0, 0);
  /* default for firstMaximum is 0,
   * so this bool is False is no firstMaximum is given */
  bool considerOtherMaximum = std::get<0>(firstMaximum) != 0;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        /* if bigger value check if angle in range if firstMaximum is given */
        if (grid[i][j][k] > std::get<0>(maximum)) {
          auto possibleMaximum = std::make_tuple(grid[i][j][k], i, j, k);
          if (considerOtherMaximum) {
            double angle = this->calcAngleBetweenHGridPos(possibleMaximum, firstMaximum);
            if (this->idealWaterAngle_ - 5 < angle &&
                angle < this->idealWaterAngle_ + 5) {
              maximum = possibleMaximum;
            }
          } else {
            maximum = possibleMaximum;
          }
        }
        /* if equal and already firstMaximum, take better angle */
        else if (considerOtherMaximum && grid[i][j][k] == std::get<0>(maximum)) {
          double angle = calcAngleBetweenHGridPos(maximum, firstMaximum);
          auto possibleMaximum = std::make_tuple(grid[i][j][k], i, j, k);
          double newAngle = this->calcAngleBetweenHGridPos(possibleMaximum, firstMaximum);
          if (std::fabs(newAngle - this->idealWaterAngle_)
              < std::fabs(angle - this->idealWaterAngle_))
            maximum = possibleMaximum;
          }
        }
      }
    }
  }
  return maximum;
}

/**
 * @brief calculate angle in degrees between two hGrid points
 *
 * @argument a Point in the grid
 * @argument b Point in the grid
 *
 * @return double angle in degrees
 */
double Action_GIGist::calcAngleBetweenHGridPos(const std::tuple<int, int, int, int>& a, const std::tuple<int, int, int, int>& b) const {
  double xa = (std::get<1>(a) - 10) * 0.1;
  double ya = (std::get<2>(a) - 10) * 0.1;
  double za = (std::get<3>(a) - 10) * 0.1;
  double xb = (std::get<1>(b) - 10) * 0.1;
  double yb = (std::get<2>(b) - 10) * 0.1;
  double zb = (std::get<3>(b) - 10) * 0.1;
  double dotProduct = (xa * xb + ya * yb + za * zb) / std::sqrt( (xa * xa + ya * ya + za * za) * (xb * xb + yb * yb + zb * zb) );
  double angle =  180 * acos(dotProduct) / Constants::PI;
  return angle;
}

/**
 * @brief sets hGrid points within 0.5 A of given maximum to 0
 *
 * @argument grid The grid storing the values
 * @argument dim The dimension of the grid
 * @argument maximum The point around which points will be deleted
 */
void Action_GIGist::deleteAroundFirstH(std::vector<std::vector<std::vector<int>>>& grid, const int dim, const std::tuple<int, int, int, int>& maximum) const {
  double destroyDistance = 0.5;
  double destroyDistanceSq = destroyDistance * destroyDistance;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        std::tuple<int, int, int, int> position = std::make_tuple(0, i, j, k);
        double distance = calcDistSqBetweenHGridPos(position, maximum);
        if (distance <= destroyDistanceSq)
          grid[i][j][k] = 0;
      }
    }
  }
}

/**
 * @brief calculate squared distance between two hGrid points
 *
 * @argument a Point in the grid
 * @argument b Point in the grid
 *
 * @return double squared distance
 */
double Action_GIGist::calcDistSqBetweenHGridPos(const std::tuple<int, int, int, int>& a, const std::tuple<int, int, int, int>& b) const {
  double xa = (std::get<1>(a) - 10) * 0.1;
  double ya = (std::get<2>(a) - 10) * 0.1;
  double za = (std::get<3>(a) - 10) * 0.1;
  double xb = (std::get<1>(b) - 10) * 0.1;
  double yb = (std::get<2>(b) - 10) * 0.1;
  double zb = (std::get<3>(b) - 10) * 0.1;
  double distance = (xa - xb) * (xa - xb) + (ya - yb) * (ya - yb) + (za - zb) * (za - zb);
  return distance;
}

/**
 * @brief gives cartesian coordinates of point in hGrid
 *
 * @argument pos Point in grid
 *
 * @return Vec3 coordinates of the point
 */
Vec3 Action_GIGist::coordsFromHGridPos(const std::tuple<int, int, int, int>& pos) const {
  double x = (std::get<1>(pos) - 10) * 0.1;
  double y = (std::get<2>(pos) - 10) * 0.1;
  double z = (std::get<3>(pos) - 10) * 0.1;
  return Vec3(x, y, z);
}

/**
 * @brief weights Delta G of GIST with the water density that was subtracted
 *        to place the water molecule
 *
 * @argument index The index where the oxygen is placed
 * @argument shellNum The number of shells around the placed oxygen to add to the density
 * @argument densityValue The value after the last shell was added
 * @argument densityValueOld The value before the last shell was added
 * @argument relPop The list of the population values
 * @argument deltaG The list of all DeltaG values
 *
 * @return double density weighted Delta G
 */
double Action_GIGist::assignDensityWeightedDeltaG(
  int index,
  int shellNum,
  double densityValue,
  double densityValueOld,
  const std::vector<double>& relPop,
  const std::vector<double>& deltaG
) {
  double value = 0.0; // value to be returned
  /* cycle through all shells but the last one */
  for (int i = 0; i < shellNum; ++i) {
    /* current shell */
    auto shell = std::make_shared<std::vector<int>>(
        shellcontainer_[shellcontainerKeys_[i]]);
    /* cycle through current shell */
    for (unsigned int j = 0; j < (*shell).size(); ++j) {
      /* get index and check if inside the grid */
      int tmpIndex = index + (*shell)[j];
      if (0 < tmpIndex && tmpIndex < static_cast<int>(info_.grid.nVoxels))
        /* add density weighted delta G to value */
        value += relPop[tmpIndex] * deltaG[tmpIndex];
    }
  }
  double last_shell = densityValue - densityValueOld; // density of last shell
  /* Get percentage of how much of the last shell shall be accounted for */
  double percentage = 1.0;
  if (last_shell != 0.0)
    percentage -= (densityValue - 1 / (info_.grid.voxelVolume * info_.system.rho0)) / last_shell;
  /* identical to above but only last shell and percentage */
  auto outerShell = std::make_shared<std::vector<int>>(
      shellcontainer_[shellcontainerKeys_[shellNum]]);
  for (unsigned int i = 0; i < (*outerShell).size(); ++i) {
    int tmpIndex = index + (*outerShell)[i];
    if (0 < tmpIndex && tmpIndex < static_cast<int>(info_.grid.nVoxels))
      value += percentage * relPop[tmpIndex] * deltaG[tmpIndex];
  }
  return value * info_.grid.voxelVolume * info_.system.rho0;
}

/**
 * @brief adds density of additional water shell to the densityValue
 *
 * @argument densityValue The value after the last shell was added
 * @argument relPop The list of the population values
 * @argument index The index of the GIST grid where the oxygen will be placed
 * @argument shellNum The number of the new shells to be added
 *
 * @return double density value with the new water shell
 */
double Action_GIGist::addWaterShell(double& densityValue, const std::vector<double>& relPop, const int index, const int shellNum) {
  /* get shell from map */
  auto newShell = std::make_shared<std::vector<int>>(
      shellcontainer_[shellcontainerKeys_[shellNum]]
  );
  for (unsigned int i = 0; i < (*newShell).size(); ++i) {
    int tmpIndex = index + (*newShell)[i];
    if (0 < tmpIndex && tmpIndex < static_cast<int>(info_.grid.nVoxels))
      densityValue += relPop[tmpIndex];
  }
  return densityValue;
}

/**
 * @brief subtract density from all voxels that belonged to the included shells
 *
 * @argument relPop The list of the population values
 * @argument index The index where the oxygen is placed
 * @argument shellNum The number of shells around the placed oxygen that were included
 * @argument densityValue The value after the last shell was added
 * @argument densityValueOld The value before the last shell was added
 */
void Action_GIGist::subtractWater
(
  std::vector<double>& relPop,
  int index,
  int shellNum,
  double densityValue,
  double densityValueOld
) 
{
  /* cycle through all but the last shell */
  for (int i = 0; i < shellNum; ++i) {
    auto shell = std::make_shared<std::vector<int>>(
        shellcontainer_[shellcontainerKeys_[i]]);
    for (unsigned int j = 0; j < (*shell).size(); ++j) {
      int tmpIndex = index + (*shell)[j];
      if (0 < tmpIndex && tmpIndex < static_cast<int>(info_.grid.nVoxels))
        /* remove all population from the voxel in the GIST grid */
        relPop[tmpIndex] = 0.0;
    }
  }
  /* since density of one water is overshot, the density must not be deleted
   * completely in the last shell, first determine percentage */
  double last_shell = densityValue - densityValueOld; // density in last shell
  double percentage = 1.0;
  if (last_shell != 0.0)
    percentage -= (densityValue - 1 / (info_.grid.voxelVolume * info_.system.rho0)) / last_shell;
  /* identical to before but only last shell and percentage */
  auto outerShell = std::make_shared<std::vector<int>>(
      shellcontainer_[shellcontainerKeys_[shellNum]]);
  for (unsigned int i = 0; i < (*outerShell).size(); ++i) {
    int tmpIndex = index + (*outerShell)[i];
    if (0 < tmpIndex && tmpIndex < static_cast<int>(info_.grid.nVoxels))
      relPop[tmpIndex] -= percentage * relPop[tmpIndex];
  }
}

/**
 * @brief writes placed water into pdb
 *
 * @argument atomNumber The running index in the pdb file
 * @argument voxelCoords The coordinates for the oxygen
 * @argument h1 coordinates of one hydrogen relative to the oxygen
 * @argument h2 coordinates of the other hydrogen relative to the oxygen
 * @argument deltaG density weighted Delta G to be included as b-factor
 */
void Action_GIGist::writeFebissPdb
(
  const int atomNumber,
  const Vec3& voxelCoords,
  const Vec3& h1,
  const Vec3& h2,
  const double deltaG
) {
  /* get absolute coordinates of hydrogens */
  Vec3 h1Coords = h1 + voxelCoords;
  Vec3 h2Coords = h2 + voxelCoords;

  febissWaterfile_->Printf(
    "HETATM%5d    O FEB     1    %8.3f%8.3f%8.3f%6.2f%7.2f           O  \n",
    atomNumber+1,
    voxelCoords[0],
    voxelCoords[1],
    voxelCoords[2],
    1.00,
    deltaG
  );

  febissWaterfile_->Printf(
    "HETATM%5d    H FEB     1    %8.3f%8.3f%8.3f%6.2f%7.2f           H  \n",
    atomNumber+2,
    h1Coords[0],
    h1Coords[1],
    h1Coords[2],
    1.00,
    deltaG
  );

  febissWaterfile_->Printf(
    "HETATM%5d    H FEB     1    %8.3f%8.3f%8.3f%6.2f%7.2f           H  \n",
    atomNumber+3,
    h2Coords[0],
    h2Coords[1],
    h2Coords[2],
    1.00,
    deltaG
  );
}
