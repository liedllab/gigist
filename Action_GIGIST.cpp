#include "Action_GIGIST.h"


/**
 * Standard constructor
 */
Action_GIGist::Action_GIGist() :
#ifdef CUDA
NBindex_c_(NULL),
molecule_c_(NULL),
paramsLJ_c_(NULL),
max_c_(NULL),
min_c_(NULL),
result_w_c_(NULL),
result_s_c_(NULL),
result_O_c_(NULL),
result_N_c_(NULL),
#endif
list_(NULL),
rho0_(0),
numberSolvent_(0),
numberAtoms_(0),
temperature_(0),
nFrames_(0),
forceStart_(-1),
centerSolventIdx_(0),
headAtomType_(0),
top_(NULL),
voxelVolume_(0),
voxelSize_(0),
nVoxels_(0),
dimensions_(Vec3()),
center_(Vec3()),
gridStart_(Vec3()),
gridEnd_(Vec3()),
solvent_(NULL),
dict_(DataDictionary()),
datafile_(NULL),
dxfile_(NULL),
writeDx_(false),
doorder_(false)
{}

/**
 * The help function.
 */
void Action_GIGist::Help() const {
  mprintf("     Usage:\n"
          "    griddim [dimx dimy dimz]   Defines the dimension of the grid.\n"
          "    <temp 300>                 Defines the temperature of the simulation.\n"
          "    <gridspacn 0.5>            Defines the grid spacing\n"
          "    <refdens 0.0329>           Defines the reference density for the water model.\n"
          "    <out \"out.dat\">            Defines the name of the output file.\n" 
          "    <dx>                       Set to write out dx files. Population is always written.\n"

          "  The griddimensions must be set in integer values and have to be larger than 0.\n"
          "  The greatest advantage, stems from the fact that this code is parallelized\n"
          "  on the GPU.\n\n"

          "  The code is meant to run on the GPU. Therefore, the CPU implementation of GIST\n"
          "  in this code is probably slower than the original GIST implementation.\n\n"

          "  When using this GIST implementation please cite:\n"
          "#    Johannes Kraml, Anna S. Kamenik, Franz Waibl, Michael Schauperl, Klaus R. Liedl, JCTC (2019), ACCEPTED\n"
          "#    Steven Ramsey, Crystal Nguyen, Romelia Salomon-Ferrer, Ross C. Walker, Michael K. Gilson, and Tom Kurtzman\n"
          "#      J. Comp. Chem. 37 (21) 2016\n"
          "#    Crystal Nguyen, Michael K. Gilson, and Tom Young, arXiv:1108.4876v1 (2011)\n"
          "#    Crystal N. Nguyen, Tom Kurtzman Young, and Michael K. Gilson,\n"
          "#      J. Chem. Phys. 137, 044101 (2012)\n"
          "#    Lazaridis, J. Phys. Chem. B 102, 3531–3541 (1998)\n");
}

Action_GIGist::~Action_GIGist() {
  delete[] this->solvent_;
}

/**
 * Initialize the GIST calculation by setting up the users input.
 * @argument argList: The argument list of the user.
 * @argument actionInit: The action initialization object.
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

  this->temperature_ = argList.getKeyDouble("temp", 300.0);
  this->voxelSize_ = argList.getKeyDouble("gridspacn", 0.5);
  this->voxelVolume_ = this->voxelSize_ * this->voxelSize_ * this->voxelSize_;
  this->rho0_ = argList.getKeyDouble("refdens", 0.0329);
  this->forceStart_ = argList.getKeyInt("force", -1);
  this->neighbourCut2_ = argList.getKeyDouble("neighbour", 3.5);
  this->neighbourCut2_ *= this->neighbourCut2_;
  this->nFrames_ = 0;
  this->image_.InitImaging( true );



  if (argList.hasKey("griddim")) {
    double x = argList.getNextInteger(-1);
    double y = argList.getNextInteger(-1);
    double z = argList.getNextInteger(-1);
    if ( (x < 0) || (y < 0) || (z < 0) ) {
      mprinterr("Error: Negative Values for griddimensions not allowed.\n\n");
      return Action::ERR;
    }
    this->dimensions_.SetVec(x, y, z);
  } else {
    mprinterr("Error: Dimensions must be set!\n\n");
    return Action::ERR;
  }

   if (argList.hasKey("gridcntr")) {
    double x = argList.getNextDouble(-1);
    double y = argList.getNextDouble(-1);
    double z = argList.getNextDouble(-1);
    this->center_.SetVec(x, y ,z);
  } else {
    mprintf("Warning: No grid center specified, defaulting to origin!\n\n");
    this->center_.SetVec(0, 0, 0);
  }




  if (argList.hasKey("dx")) {
    this->writeDx_ = true;
  }

  if (argList.hasKey("doorder")) {
    this->doorder_ = true;
  }


  if (argList.hasKey("out")) {
    this->datafile_ = actionInit.DFL().AddCpptrajFile( argList.GetStringNext(), "GIST output" );
  } else {
    this->datafile_ = actionInit.DFL().AddCpptrajFile( "out.dat", "GIST output" );
  }

  this->gridStart_.SetVec(this->center_[0] - (this->dimensions_[0] * 0.5) * this->voxelSize_, 
                          this->center_[1] - (this->dimensions_[1] * 0.5) * this->voxelSize_,
                          this->center_[2] - (this->dimensions_[2] * 0.5) * this->voxelSize_);
  
  this->gridEnd_.SetVec(this->center_[0] + this->dimensions_[0] * this->voxelSize_, 
                        this->center_[1] + this->dimensions_[1] * this->voxelSize_,
                        this->center_[2] + this->dimensions_[2] * this->voxelSize_);

  
  this->nVoxels_ = (unsigned int)this->dimensions_[0] * (unsigned int)this->dimensions_[1] * (unsigned int)this->dimensions_[2];
  this->quaternions_.resize(this->nVoxels_);
  this->waterCoordinates_.resize(this->nVoxels_);

  std::string dsname = actionInit.DSL().GenerateDefaultName("GIST");
  this->result_ = std::vector<DataSet_3D *>(this->dict_.size());
  for (unsigned int i = 0; i < this->dict_.size(); ++i) {
    this->result_.at(i) = (DataSet_3D*)actionInit.DSL().AddSet(DataSet::GRID_FLT, MetaData(dsname, this->dict_.getElement(i)));
    this->result_.at(i)->Allocate_N_C_D(this->dimensions_[0], this->dimensions_[1], this->dimensions_[2], 
                                          this->center_, this->voxelSize_);

    if (
        ( this->writeDx_ &&
          this->dict_.getElement(i).compare("Eww") != 0 && 
          this->dict_.getElement(i).compare("Esw") != 0 &&
          this->dict_.getElement(i).compare("dipole_xtemp") != 0 && 
          this->dict_.getElement(i).compare("dipole_ytemp") != 0 &&
          this->dict_.getElement(i).compare("dipole_ztemp") != 0 &&
          this->dict_.getElement(i).compare("order") != 0 && 
          this->dict_.getElement(i).compare("neighbour") != 0 ) ||
        i == 0 
       ) 
    {
      DataFile *file = actionInit.DFL().AddDataFile(this->dict_.getElement(i) + ".dx");
      file->AddDataSet(this->result_.at(i));
    }
  }

  mprintf("Center: %.1g %.1g %.1g, Dimensions %d %d %d\n"
          "  When using this GIST implementation please cite:\n"
          "#    Johannes Kraml, Anna S. Kamenik, Franz Waibl, Michael Schauperl, Klaus R. Liedl, JCTC (2019)\n"
          "#    Steven Ramsey, Crystal Nguyen, Romelia Salomon-Ferrer, Ross C. Walker, Michael K. Gilson, and Tom Kurtzman\n"
          "#      J. Comp. Chem. 37 (21) 2016\n"
          "#    Crystal Nguyen, Michael K. Gilson, and Tom Young, arXiv:1108.4876v1 (2011)\n"
          "#    Crystal N. Nguyen, Tom Kurtzman Young, and Michael K. Gilson,\n"
          "#      J. Chem. Phys. 137, 044101 (2012)\n"
          "#    Lazaridis, J. Phys. Chem. B 102, 3531–3541 (1998)\n",
          this->center_[0], this->center_[1], this->center_[2],
          (int)this->dimensions_[0], (int)this->dimensions_[1], (int)this->dimensions_[2]);
  
  return Action::OK;
}

/**
 * Setup for the GIST calculation. Does everything involving the Topology file.
 * @argument setup: The setup object of the cpptraj code libraries.
 * @return: Action::OK on success, Action::ERR otherwise.
 */
Action::RetType Action_GIGist::Setup(ActionSetup &setup) {
  this->solventAtomCounter_ = std::vector<int>();
  // Setup imaging and topology parsing.
  this->image_.SetupImaging( setup.CoordInfo().TrajBox().Type() );

  // Save topology and topology related values
  this->top_             = setup.TopAddress();
  this->numberAtoms_     = setup.Top().Natom();
  this->numberSolvent_   = setup.Top().Nsolvent();
  this->solvent_ = new bool[this->numberAtoms_];
  bool firstRound = true;

  // Save different values, which depend on the molecules and/or atoms.
  for (Topology::mol_iterator mol = setup.Top().MolStart(); 
          mol != setup.Top().MolEnd(); ++mol) {
    unsigned int nAtoms = (unsigned int) mol->NumAtoms();

    for (unsigned int i = 0; i < nAtoms; ++i) {
      this->molecule_.push_back( setup.Top()[mol->BeginAtom() + i].MolNum() );
      this->charges_.push_back( setup.Top()[mol->BeginAtom() + i].Charge() );
      this->atomTypes_.push_back( setup.Top()[mol->BeginAtom() + i].TypeIndex() );

      // Check if the molecule is a solvent, either by the topology parameters or because force was set.
      if ( (mol->IsSolvent() && this->forceStart_ == -1) || (( this->forceStart_ > -1 ) && ( setup.Top()[mol->BeginAtom()].MolNum() >= this->forceStart_ )) ) {
        std::string aName = setup.Top()[mol->BeginAtom() + i].ElementName();
        
        // Check if dictionary already holds an entry for the atoms name, if not add it to
        // the dictionary, if yes, add 1 to the correct solvent atom counter.
        if (! (this->dict_.contains(aName)) ) {
          this->dict_.add(aName);
          this->solventAtomCounter_.push_back(1);
        } else if (firstRound) {
          this->solventAtomCounter_.at(this->dict_.getIndex(aName) - this->result_.size()) += 1;
        }

        // Check for the centerSolventAtom (which in this easy approximation is either C or O)
        if ( this->weight(aName) < this->weight(this->centerSolventAtom_) ) {
          this->centerSolventAtom_ = setup.Top()[mol->BeginAtom() + i].ElementName();
          this->centerSolventIdx_ = i; // Assumes the same order of atoms.
          this->headAtomType_ = setup.Top()[mol->BeginAtom() + i].TypeIndex();
        }
        // Set solvent to true
        this->solvent_[mol->BeginAtom() + i] = true;
      } else {
       this->solvent_[mol->BeginAtom() + i] = false;
      }
    }
    if ((mol->IsSolvent() && this->forceStart_ == -1) || (( this->forceStart_ > -1 ) && ( setup.Top()[mol->BeginAtom()].MolNum() >= this->forceStart_ ))) {
      firstRound = false;
    }
  }
  // Add results for the different solvent atoms.
  for (unsigned int i = 0; i < (this->dict_.size() - this->result_.size()); ++i) {
    this->resultV_.push_back(std::vector<double>(this->dimensions_[0] * this->dimensions_[1] * this->dimensions_[2]));
  }

  
  // Define different things for the case that this was compiled using CUDA
#ifdef CUDA
  NonbondParmType nb = setup.Top().Nonbond();
  this->NBIndex_ = nb.NBindex();
  this->numberAtomTypes_ = nb.Ntypes();
  for (unsigned int i = 0; i < nb.NBarray().size(); ++i) {
    this->lJParamsA_.push_back( (float) nb.NBarray().at(i).A() );
    this->lJParamsB_.push_back( (float) nb.NBarray().at(i).B() );
  }

  try {
    allocateCuda(((void**)&this->NBindex_c_), this->NBIndex_.size() * sizeof(int));
    allocateCuda((void**)&this->max_c_, 3 * sizeof(float));
    allocateCuda((void**)&this->min_c_, 3 * sizeof(float));
    allocateCuda((void**)&this->result_w_c_, this->numberAtoms_ * sizeof(float));
    allocateCuda((void**)&this->result_s_c_, this->numberAtoms_ * sizeof(float));
    allocateCuda((void**)&this->result_O_c_, this->numberAtoms_ * 4 * sizeof(int));
    allocateCuda((void**)&this->result_N_c_, this->numberAtoms_ * sizeof(int));
  } catch (CudaException &e) {
    mprinterr("Error: Could not allocate memory on GPU!\n");
    this->freeGPUMemory();
    return Action::ERR;
  }
  try {
    this->copyToGPU();
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
 * @argument frameNum: The number of the frame.
 * @argument frame: The frame itself.
 * @return: Action::ERR on error, Action::OK if everything ran smoothly.
 */
Action::RetType Action_GIGist::DoAction(int frameNum, ActionFrame &frame) {
  this->nFrames_++;
  std::vector<DOUBLE_O_FLOAT> eww_result(this->numberAtoms_);
  std::vector<DOUBLE_O_FLOAT> esw_result(this->numberAtoms_);
  std::vector<std::vector<int> > order_indices;


  // CUDA necessary information
  #ifdef CUDA
  this->tEnergy_.Start();

  Matrix_3x3 ucell_m, recip_m;
  float *recip = NULL;
  float *ucell = NULL;
  int boxinfo;

  // Check Boxinfo and write the necessary data into recip, ucell and boxinfo.
  switch(this->image_.ImageType()) {
    case NONORTHO:
      recip = new float[9];
      ucell = new float[9];
      frame.Frm().BoxCrd().ToRecip(ucell_m, recip_m);
      for (int i = 0; i < 9; ++i) {
        ucell[i] = (float) ucell_m.Dptr()[i];
        recip[i] = (float) recip_m.Dptr()[i];
      }
      boxinfo = 2;
      break;
    case ORTHO:
      recip = new float[9];
      for (int i = 0; i < 3; ++i) {
        recip[i] = (float) frame.Frm().BoxCrd()[i];
      }
      ucell = NULL;
      boxinfo = 1;
      break;
    case NOIMAGE:
      recip = NULL;
      ucell = NULL;
      boxinfo = 0;
      break;
    default:
      mprinterr("Error: Unexpected box information found.");
      return Action::ERR;
  }

  std::vector<int> result_o = std::vector<int>(4 * this->numberAtoms_);
  std::vector<int> result_n = std::vector<int>(this->numberAtoms_);
  // TODO: Switch things around a bit and move the back copying to the end of the calculation.
  //       Then the time needed to go over all waters and the calculations that come with that can
  //			 be hidden quite nicely behind the interaction energy calculation.
  // Must create arrays from the vectors, does that by getting the address of the first element of the vector.
  std::vector<std::vector<float> > e_result = doActionCudaEnergy(frame.Frm().xAddress(), this->NBindex_c_, this->numberAtomTypes_, this->paramsLJ_c_, this->molecule_c_, boxinfo, recip, ucell, this->numberAtoms_, this->min_c_, 
                                                    this->max_c_, this->headAtomType_,this->neighbourCut2_, &(result_o[0]), &(result_n[0]), this->result_w_c_, 
                                                    this->result_s_c_, this->result_O_c_, this->result_N_c_, this->doorder_);
  eww_result = e_result.at(0);
  esw_result = e_result.at(1);

  if (this->doorder_) {
    int counter = 0;
    for (unsigned int i = 0; i < (4 * this->numberAtoms_); i += 4) {
      ++counter;
      std::vector<int> temp;
      for (unsigned int j = 0; j < 4; ++j) {
        temp.push_back(result_o.at(i + j));
      }
      order_indices.push_back(temp);
    }
  }

  delete[] recip;
  delete[] ucell;

  this->tEnergy_.Stop();

  #endif


  std::vector<bool> onGrid(this->top_->Natom());
  for (unsigned int i = 0; i < onGrid.size(); ++i) {
    onGrid.at(i) = false;
  }

  std::vector<Vec3> atomCoords(this->nVoxels_);

  #if defined _OPENMP && defined CUDA
  this->tHead_.Start();
  #pragma omp parallel for
  #endif
  for (Topology::mol_iterator mol = this->top_->MolStart(); mol < this->top_->MolEnd(); ++mol) {
    if ((mol->IsSolvent() && this->forceStart_ == -1) || (( this->forceStart_ > -1 ) && ( this->top_->operator[](mol->BeginAtom()).MolNum() >= this->forceStart_ ))) {
      
    
      int headAtomIndex = -1;
      // Keep voxel at -1 if it is not possible to put iy on the grid
      int voxel = -1;
      std::vector<Vec3> molAtomCoords;
      #if !defined _OPENMP && !defined CUDA
      this->tHead_.Start();
      #endif
      for (int atom1 = mol->BeginAtom(); atom1 < mol->EndAtom(); ++atom1) {
        size_t bin_i, bin_j, bin_k;
        bool first = true;
        if (this->solvent_[atom1]) { // Do we need that?
          // Save coords for later use.
          const double *vec = frame.Frm().XYZ(atom1);
          molAtomCoords.push_back(Vec3(vec));
          // Check if atom is "Head" atom of the solvent
          // Could probably save some time here by writing head atom indices into an array.
          // TODO: When assuming fixed atom position in topology, should be very easy.
          if ( std::string((*this->top_)[atom1].ElementName()).compare(this->centerSolventAtom_) == 0 && first ) {
            // Try to bin atom1 onto the grid. If it is possible, get the index and keep working,
            // if not, calculate the energies between all atoms to this point.
            if ( this->result_.at(this->dict_.getIndex("population"))->Bin().Calc(vec[0], vec[1], vec[2], bin_i, bin_j, bin_k) ) {
              voxel = this->result_.at(this->dict_.getIndex("population"))->CalcIndex(bin_i, bin_j, bin_k);
              headAtomIndex = atom1 - mol->BeginAtom();
              atomCoords.at(voxel) = Vec3(vec[0], vec[1], vec[2]);
              // Does not necessarily need this
              this->waterCoordinates_.at(voxel).push_back(frame.Frm().XYZ(atom1));
              // Does not necessarily need this
              this->resultV_.at(this->dict_.getIndex(this->centerSolventAtom_) - this->result_.size()).at(voxel) += 1.0;
              
              #if !defined _OPENMP && !defined CUDA
              this->tDipole_.Start();
              #endif
              double DPX = 0;
              double DPY = 0;
              double DPZ = 0;
              for (int atoms = mol->BeginAtom(); atoms < mol->EndAtom(); ++atoms) {
                onGrid.at(atoms) = true;
                const double *XYZ = frame.Frm().XYZ(atoms);
                DPX += this->top_->operator[](atoms).Charge() * XYZ[0];
                DPY += this->top_->operator[](atoms).Charge() * XYZ[1];
                DPZ += this->top_->operator[](atoms).Charge() * XYZ[2];
              }
              this->result_.at(this->dict_.getIndex("dipole_xtemp"))->UpdateVoxel(voxel, DPX);
              this->result_.at(this->dict_.getIndex("dipole_ytemp"))->UpdateVoxel(voxel, DPY);
              this->result_.at(this->dict_.getIndex("dipole_ztemp"))->UpdateVoxel(voxel, DPZ);
              #if !defined _OPENMP && !defined CUDA
              this->tDipole_.Stop();
              #endif
            }
            first = false;
          } else {
            if ( this->result_.at(this->dict_.getIndex("population"))->Bin().Calc(vec[0], vec[1], vec[2], bin_i, bin_j, bin_k) ) {
              std::string aName = this->top_->operator[](atom1).ElementName();
              int voxTemp = this->result_.at(this->dict_.getIndex("population"))->CalcIndex(bin_i, bin_j, bin_k);
              this->resultV_.at(this->dict_.getIndex(aName) - this->result_.size()).at(voxTemp) += 1.0;
            }
          }
        }
      }
      #if !defined _OPENMP && !defined CUDA
      this->tHead_.Stop();
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
              Y.Normalize();
              setY = true;
            }
            if (!setX) {
              X.SetVec(molAtomCoords.at(i)[0] - molAtomCoords.at(headAtomIndex)[0], 
                        molAtomCoords.at(i)[1] - molAtomCoords.at(headAtomIndex)[1], 
                        molAtomCoords.at(i)[2] - molAtomCoords.at(headAtomIndex)[2]);
              X.Normalize();
              setX = true;
            }
            if (setX && setY) {
              break;
            }
          }
        }

        
        // Create Quaternion for the rotation from the new coordintate system to the lab coordinate system.
        Quaternion<DOUBLE_O_FLOAT> quat(X, Y);
        // The Quaternion would create the rotation of the lab coordinate system onto the
        // calculated solvent coordinate system. The invers quaternion is exactly the rotation of
        // the solvent coordinate system onto the lab coordinate system.
        quat.invert();
        this->quaternions_.at(voxel).push_back(quat);

        #if !defined _OPENMP && !defined CUDA
        this->tRot_.Stop();
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
        if (this->doorder_) {
          double sum = 0;
          Vec3 cent( frame.Frm().xAddress() + (mol->BeginAtom() + headAtomIndex) * 3 );
          std::vector<Vec3> vectors;
          switch(this->image_.ImageType()) {
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
              double cosThet = (vectors.at(i) * vectors.at(j)) / sqrt(vectors.at(i).Magnitude2() * vectors.at(j).Magnitude2());
              sum += (cosThet + 1.0/3) * (cosThet + 1.0/3);
            }
          }
        
          this->result_.at(this->dict_.getIndex("order"))->UpdateVoxel(voxel, 1.0 - (3.0/8.0) * sum);
        }
        this->result_.at(this->dict_.getIndex("neighbour"))->UpdateVoxel(voxel, result_n.at(mol->BeginAtom() + headAtomIndex));
        // End of calculation of the order parameters

        #ifndef _OPENMP
        this->tEadd_.Start();
        #endif
        // There is absolutely nothing to check here, as the solute can not be in place here.
        for (int atom = mol->BeginAtom(); atom < mol->EndAtom(); ++atom) {
          // Just adds up all the interaction energies for this voxel.
          this->result_.at(this->dict_.getIndex("Eww"))->UpdateVoxel(voxel, (double)eww_result.at(atom));
          this->result_.at(this->dict_.getIndex("Esw"))->UpdateVoxel(voxel, (double)esw_result.at(atom));
        }
        #ifndef _OPENMP
        this->tEadd_.Stop();
        #endif
  #endif
      }

      // If CUDA is used, energy calculations are already done.
  #ifndef CUDA
      if (voxel != -1 ) {
        std::vector<Vec3> nearestWaters(4);
        // Use HUGE distances at the beginning. This is defined as 3.40282347e+38F.
        double distances[4] = {HUGE, HUGE, HUGE, HUGE};
        // Needs to be fixed, one does not need to calculate all interactions each time.
        for (int atom1 = mol->BeginAtom(); atom1 < mol->EndAtom(); ++atom1) {
          double eww = 0;
          double esw = 0;
  // OPENMP only over the inner loop

  #if defined _OPENMP
          #pragma omp parallel for
  #endif
          for (unsigned int atom2 = 0; atom2 < this->numberAtoms_; ++atom2) {
            if ( (*this->top_)[atom1].MolNum() != (*this->top_)[atom2].MolNum() ) {
              this->tEadd_.Start();
              double r_2 = this->calcDistanceSqrd(frame, atom1, atom2);
              double energy = this->calcEnergy(r_2, atom1, atom2);
              this->tEadd_.Stop();
              if (this->solvent_[atom2]) {
  #ifdef _OPENMP
                #pragma omp atomic
  #endif
                eww += energy;
              } else {
  #ifdef _OPENMP
                #pragma omp atomic
  #endif
                esw += energy;
              }
              if (this->atomTypes_.at(atom1) == this->headAtomType_ &&
                  this->atomTypes_.at(atom2) == this->headAtomType_) {
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
                if (r_2 < this->neighbourCut2_) {
                  this->result_.at(this->dict_.getIndex("neighbour"))->UpdateVoxel(voxel, 1);
                }
              }
            }
          }
          double sum = 0;
          for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 4; ++j) {
              double cosThet = (nearestWaters.at(i) * nearestWaters.at(j)) / 
                                sqrt(nearestWaters.at(i).Magnitude2() * nearestWaters.at(j).Magnitude2());
              sum += (cosThet + 1.0/3) * (cosThet + 1.0/3);
            }
          }
          this->result_.at(this->dict_.getIndex("order"))->UpdateVoxel(voxel, 1.0 - (3.0/8.0) * sum);
          eww /= 2.0;
          this->result_.at(this->dict_.getIndex("Eww"))->UpdateVoxel(voxel, eww);
          this->result_.at(this->dict_.getIndex("Esw"))->UpdateVoxel(voxel, esw);
        }
      }
#endif
    }
  }
  #if defined _OPENMP && defined CUDA
  this->tHead_.Stop();
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
   */
  #ifdef CUDA_UPDATED
  std::vector<std::vector<float> > dTSTest = doActionCudaEntropy(this->waterCoordinates_, this->dimensions_[0], this->dimensions_[1],
                                                              this->dimensions_[2], this->quaternions_, this->temperature_, this->rho0_, this->nFrames_);
  #endif
  mprintf("Processed %d frames.\nMoving on to entropy calculation.\n", this->nFrames_);
  ProgressBar progBarEntropy(this->nVoxels_);
#ifdef _OPENMP
  int curVox = 0;
  #pragma omp parallel for
#endif
  for (unsigned int voxel = 0; voxel < this->nVoxels_; ++voxel) {
    // If _OPENMP is defined, the progress bar has to be updated critically,
    // to ensure the right addition.
#ifndef _OPENMP
    progBarEntropy.Update( voxel );
#else
    #pragma omp critical
    progBarEntropy.Update( curVox++ );
#endif
    double dTSorient_norm   = 0;
    double dTStrans_norm    = 0;
    double dTSsix_norm      = 0;
    double dTSorient_dens   = 0;
    double dTStrans_dens    = 0;
    double dTSsix_dens      = 0;
    double Esw_norm         = 0;
    double Esw_dens         = 0;
    double Eww_norm         = 0;
    double Eww_dens         = 0;
    double order_norm       = 0;
    double neighbour_dens   = 0;
    double neighbour_norm   = 0;
    // Only calculate if there is actually water molecules at that position.
    if (this->result_.at(this->dict_.getIndex("population"))->operator[](voxel) > 0) {
      double pop = this->result_.at(this->dict_.getIndex("population"))->operator[](voxel);
      #ifdef CUDA_UPDATED
      dTSorient_norm          = dTSTest.at(1).at(voxel);
      dTStrans_norm           = dTSTest.at(0).at(voxel);
      dTSsix_norm             = dTSTest.at(2).at(voxel);
      #else
      dTSorient_norm          = this->calcOrientEntropy(voxel);
      std::vector<double> dTS = this->calcTransEntropy(voxel);
      dTStrans_norm           = dTS.at(0);
      dTSsix_norm             = dTS.at(1);
      #endif
      dTSorient_dens          = dTSorient_norm * pop / (this->nFrames_ * this->voxelVolume_);
      dTStrans_dens           = dTStrans_norm * pop / (this->nFrames_ * this->voxelVolume_);
      dTSsix_dens             = dTSsix_norm * pop / (this->nFrames_ * this->voxelVolume_);
      
      Esw_norm = this->result_.at(this->dict_.getIndex("Esw"))->operator[](voxel) / pop;
      Esw_dens = this->result_.at(this->dict_.getIndex("Esw"))->operator[](voxel) / (this->nFrames_ * this->voxelVolume_);
      Eww_norm = this->result_.at(this->dict_.getIndex("Eww"))->operator[](voxel) / pop;
      Eww_dens = this->result_.at(this->dict_.getIndex("Eww"))->operator[](voxel) / (this->nFrames_ * this->voxelVolume_);
      order_norm = this->result_.at(this->dict_.getIndex("order"))->operator[](voxel) / pop;
      neighbour_norm = this->result_.at(this->dict_.getIndex("neighbour"))->operator[](voxel) / pop;
      neighbour_dens = this->result_.at(this->dict_.getIndex("neighbour"))->operator[](voxel) / (this->nFrames_ * this->voxelVolume_);
    }
    
    // Calculate the final dipole values. The temporary data grid has to be used, as data
    // already saved cannot be updated.
    double DPX = this->result_.at(this->dict_.getIndex("dipole_xtemp"))->operator[](voxel) / (DEBYE * this->nFrames_ * this->voxelVolume_);
    double DPY = this->result_.at(this->dict_.getIndex("dipole_ytemp"))->operator[](voxel) / (DEBYE * this->nFrames_ * this->voxelVolume_);
    double DPZ = this->result_.at(this->dict_.getIndex("dipole_ztemp"))->operator[](voxel) / (DEBYE * this->nFrames_ * this->voxelVolume_);
    double DPG = sqrt( DPX * DPX + DPY * DPY + DPZ * DPZ );
    this->result_.at(this->dict_.getIndex("dTStrans_norm"))->UpdateVoxel(voxel, dTStrans_norm);
    this->result_.at(this->dict_.getIndex("dTStrans_dens"))->UpdateVoxel(voxel, dTStrans_dens);
    this->result_.at(this->dict_.getIndex("dTSorient_norm"))->UpdateVoxel(voxel, dTSorient_norm);
    this->result_.at(this->dict_.getIndex("dTSorient_dens"))->UpdateVoxel(voxel, dTSorient_dens);
    this->result_.at(this->dict_.getIndex("dTSsix_norm"))->UpdateVoxel(voxel, dTSsix_norm);
    this->result_.at(this->dict_.getIndex("dTSsix_dens"))->UpdateVoxel(voxel, dTSsix_dens);
    this->result_.at(this->dict_.getIndex("order_norm"))->UpdateVoxel(voxel, order_norm);
    this->result_.at(this->dict_.getIndex("neighbour_norm"))->UpdateVoxel(voxel, neighbour_norm);
    this->result_.at(this->dict_.getIndex("neighbour_dens"))->UpdateVoxel(voxel, neighbour_dens);

    
    this->result_.at(this->dict_.getIndex("Esw_norm"))->UpdateVoxel(voxel, Esw_norm);
    this->result_.at(this->dict_.getIndex("Esw_dens"))->UpdateVoxel(voxel, Esw_dens);
    this->result_.at(this->dict_.getIndex("Eww_norm"))->UpdateVoxel(voxel, Eww_norm);
    this->result_.at(this->dict_.getIndex("Eww_dens"))->UpdateVoxel(voxel, Eww_dens);
    // Maybe there is a better way, I have to look that
    this->result_.at(this->dict_.getIndex("dipole_x"))->UpdateVoxel(voxel, DPX);
    this->result_.at(this->dict_.getIndex("dipole_y"))->UpdateVoxel(voxel, DPY);
    this->result_.at(this->dict_.getIndex("dipole_z"))->UpdateVoxel(voxel, DPZ);
    this->result_.at(this->dict_.getIndex("dipole_g"))->UpdateVoxel(voxel, DPG);
    for (unsigned int i = 0; i < this->resultV_.size(); ++i) {
      this->resultV_.at(i).at(voxel) /= (this->nFrames_ * this->voxelVolume_ * this->rho0_ * this->solventAtomCounter_.at(i));
    }
  }

  
  mprintf("Writing output:\n");
  this->datafile_->Printf("GIST calculation output.\n");
  this->datafile_->Printf("   voxel        x          y          z         population     dTSt_d(kcal/mol)  dTSt_n(kcal/mol)"
                          "  dTSo_d(kcal/mol)  dTSo_n(kcal/mol)  dTSs_d(kcal/mol)  dTSs_n(kcal/mol)   "
                          "Esw_d(kcal/mol)   Esw_n(kcal/mol)   Eww_d(kcal/mol)   Eww_n(kcal/mol)    dipoleX    "
                          "dipoleY    dipoleZ    dipole    neighbour_d    neighbour_n    order_n  ");
  // Moved the densities to the back of the output file, so that the energies are always
  // at the same positions.
  for (unsigned int i = this->result_.size(); i < this->dict_.size(); ++i) {
    this->datafile_->Printf("  g_%s  ", this->dict_.getElement(i).c_str());
  }
  this->datafile_->Printf("\n");

// Ignore this part as it needs great amounts of space
#ifdef BEAUTIFUL_OUTPUT
  for (unsigned int voxel = 0; voxel < this->nVoxels_; ++voxel) {
    size_t i, j, k;
    this->result_.at(this->dict_.getIndex("population"))->ReverseIndex(voxel, i, j, k);
    Vec3 coords = this->result_.at(this->dict_.getIndex("population"))->Bin().Center(i, j, k);
    this->datafile_->Printf("% 11d%11.3f%11.3f%11.3f%18.0f%18.3f%18.3f%18.3f%18.3f%18.3f%18.3f%18.3f%18.3f%18.3f%18.3f\n", 
                            voxel, coords[0], coords[1], coords[2],
                            this->result_.at(this->dict_.getIndex("population"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTStrans_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTStrans_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTSorient_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTSorient_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTSsix_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTSsix_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("Esw_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("Esw_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("Eww_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("Eww_norm"))->operator[](voxel)
                            );
  }
#endif
  // Final output, the DX files are done automatically by cpptraj
  // so only the standard GIST-format is done here
  ProgressBar progBarIO(this->nVoxels_);
  for (unsigned int voxel = 0; voxel < this->nVoxels_; ++voxel) {
    progBarIO.Update( voxel );
    size_t i, j, k;
    this->result_.at(this->dict_.getIndex("population"))->ReverseIndex(voxel, i, j, k);
    Vec3 coords = this->result_.at(this->dict_.getIndex("population"))->Bin().Center(i, j, k);
    this->datafile_->Printf("%d %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g", 
                            voxel, coords[0], coords[1], coords[2],
                            this->result_.at(this->dict_.getIndex("population"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTStrans_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTStrans_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTSorient_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTSorient_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTSsix_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dTSsix_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("Esw_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("Esw_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("Eww_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("Eww_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dipole_x"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dipole_y"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dipole_z"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("dipole_g"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("neighbour_dens"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("neighbour_norm"))->operator[](voxel),
                            this->result_.at(this->dict_.getIndex("order_norm"))->operator[](voxel)
                            );
    for (unsigned int i = 0; i < this->resultV_.size(); ++i) {
      this->datafile_->Printf(" %g", this->resultV_.at(i).at(voxel));
    }
    this->datafile_->Printf("\n");
  }
  // The atom densities of the solvent compared to the reference density.
  if (this->writeDx_) {
    for (unsigned int i = 0; i < this->resultV_.size(); ++i) {
      this->writeDxFile("g_" + this->dict_.getElement(this->result_.size() + i) + ".dx", this->resultV_.at(i));
    }
  }
  mprintf("Timings:\n"
          " Find Head Atom:   %8.3f\n"
          " Add up Energy:    %8.3f\n"
          " Calculate Dipole: %8.3f\n"
          " Calculate Quat:   %8.3f\n"
          " Calculate Energy: %8.3f\n\n",
          this->tHead_.Total(),
          this->tEadd_.Total(),
          this->tDipole_.Total(),
          this->tRot_.Total(),
          this->tEnergy_.Total());
  #ifdef CUDA
  this->freeGPUMemory();
  #endif
}

/**
 * Calculate the Van der Waals and electrostatic energy.
 * @argument frm: The frame for which to calculate the energy.
 * @argument a1: The first atom.
 * @argument a2: The second atom.
 * @return: The interaction energy between the two atoms.
 */
double Action_GIGist::calcEnergy(double r_2, int a1, int a2) {
  r_2 = 1 / r_2;
  return this->calcElectrostaticEnergy(r_2, a1, a2) + this->calcVdWEnergy(r_2, a1, a2);
}

/**
 * Calculate the squared distance between two atoms.
 * @argument frm: The frame for which to calculate the distance.
 * @argument a1: The first atom for the calculation.
 * @argument a2: The second atom for the calculation.
 * @return: The squared distance between the two atoms.
 */
double Action_GIGist::calcDistanceSqrd(ActionFrame &frm, int a1, int a2) {
    Matrix_3x3 ucell, recip;
    double dist = 0;
    Vec3 vec1 = Vec3(frm.Frm().XYZ(a1));
    Vec3 vec2 = Vec3(frm.Frm().XYZ(a2));
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
 * @argument r_2: The squared distance between the atoms.
 * @argument a1: The atom index of atom 1.
 * @argument a2: The atom index of atom 2.
 * @return: The electrostatic energy.
 */
double Action_GIGist::calcElectrostaticEnergy(double r_2_i, int a1, int a2) {
    //double q1 = this->top_->operator[](a1).Charge();
    //double q2 = this->top_->operator[](a2).Charge();
    double q1 = this->charges_.at(a1);
    double q2 = this->charges_.at(a2);
    return q1 * Constants::ELECTOAMBER * q2 * Constants::ELECTOAMBER * sqrt(r_2_i);
}

/**
 * Calculate the van der Waals interaction energy between
 * two different atoms, as follows:
 * E(vdw) = A / (r ** 12) - B / (r ** 6)
 * Be aware that the inverse is used, as to calculate faster.
 * @argument r_2: The squared distance between the two atoms.
 * @argument a1: The atom index of atom1.
 * @argument a2: The atom index of atom2.
 * @return: The VdW interaction energy.
 */
double Action_GIGist::calcVdWEnergy(double r_2_i, int a1, int a2) {
    // Attention, both r_6 and r_12 are actually inverted. This is very ok, and makes the calculation faster.
    // However, it is not noted, thus it could be missleading
    double r_6 = r_2_i * r_2_i * r_2_i;
    double r_12 = r_6 * r_6;
    NonbondType const &params = this->top_->GetLJparam(a1, a2);
    return params.A() * r_12 - params.B() * r_6;
}

/**
 * Calculate the orientational entropy of the water atoms
 * in a given voxel.
 * @argument voxel: The index of the voxel.
 * @return: The entropy of the water atoms in that voxel.
 */
double Action_GIGist::calcOrientEntropy(int voxel) {
  int nwtotal = this->result_.at(this->dict_.getIndex("population"))->operator[](voxel);
  if(nwtotal < 2) {
    return 0;
  }
  double result = 0.0;
  for (int n0 = 0; n0 < nwtotal; ++n0) {
    double NNr = 100000;
    for (int n1 = 0; n1 < nwtotal; ++n1) {
      if (n1 == n0) {
        continue;
      }

      double rR = this->quaternions_.at(voxel).at(n0).distance(this->quaternions_.at(voxel).at(n1));
      if ( (rR < NNr) && (rR > 0) ) {
        NNr = rR;
      }
    }
    if (NNr < 9999 && NNr > 0) {
      result += log(NNr * NNr * NNr * nwtotal / (3.0 * Constants::TWOPI));
    }
  }
  return Constants::GASK_KCAL * this->temperature_ * (result / nwtotal + Constants::EULER_MASC);
}

/**
 * Calculate the translational entropy.
 * @argument voxel: The voxel for which to calculate the translational entropy.
 * @return: A vector type object, holding the values for the translational
 *          entropy, as well as the six integral entropy.
 */
std::vector<double> Action_GIGist::calcTransEntropy(int voxel) {
  // Will hold the two values dTStrans and dTSsix
  std::vector<double> ret;
  ret.push_back(0);
  ret.push_back(0);
  int nwtotal = (*this->result_.at(this->dict_.getIndex("population")))[voxel];
  for (int n0 = 0; n0 < nwtotal; ++n0) {
    double NNd = HUGE;
    double NNs = HUGE;
    // Self is not migrated to the function, as it would need to have a check against self comparison
    // The need is not entirely true, as it would produce 0 as a value.
    for (int n1 = 0; n1 < nwtotal; ++n1) {
      if (n1 == n0) {
        continue;
      }
      double dd = (this->waterCoordinates_.at(voxel).at(n0) - this->waterCoordinates_.at(voxel).at(n1)).Magnitude2();
      if (dd > 0 && dd < NNd) {
        NNd = dd;
      }
      double rR = this->quaternions_.at(voxel).at(n0).distance(this->quaternions_.at(voxel).at(n1));
      double ds = rR * rR + dd;
      if (ds < NNs && ds > 0) {
        NNs = ds;
      }
    }
    // Get the values for the dimensions
    Vec3 step;
    step.SetVec(this->dimensions_[2] * this->dimensions_[1], this->dimensions_[2], 1);
    int x = voxel / (this->dimensions_[2] * this->dimensions_[1]);
    int y = (voxel / (int)this->dimensions_[2]) % (int)this->dimensions_[1];
    int z = voxel % (int)this->dimensions_[2];
    // Check if the value is inside the boundaries, i.e., there is still a neighbour in
    // each direction.
    if ( !((x <= 0                       ) || (y <= 0                       ) || (z <= 0                       ) ||
           (x >= this->dimensions_[0] - 1) || (y >= this->dimensions_[1] - 1) || (z >= this->dimensions_[2] - 1)) ) {
      // Builds a 3 x 3 x 3 cube for entropy calculation

      for (int dim1 = -1; dim1 < 2; ++dim1) {
        for (int dim2 = -1; dim2 < 2; ++dim2) {
          for (int dim3 = -1; dim3 < 2; ++dim3) {
            int voxel2 = voxel + dim1 * step[0] + dim2 * step[1] + dim3 * step[2];
            // Checks if the voxel2 is the same as voxel, if so, has already been calculated
            if (voxel2 != voxel) {
              this->calcTransEntropyDist(voxel, voxel2, n0, NNd, NNs);
            }
          }
        }
      }
      // Bring NNd back to power 1. NNs is kept at power 2, since it is needed as a power 3 later.
      NNd = sqrt(NNd);
    } else {
      ret.at(0) = 0;
      ret.at(1) = 0;
      return ret;
    }
    
    if (NNd < 3 && NNd > 0) {
      // For both, the number of frames is used as the number of measurements.
      // The third power of NNd has to be taken, since NNd is only power 1.
      ret.at(0) += log(NNd * NNd * NNd * this->nFrames_ * 4 * Constants::PI * this->rho0_ / 3.0);
      // NNs is used to the power of 6, since it is already power of 2, only the third power
      // has to be calculated.
      ret.at(1) += log(NNs * NNs * NNs * this->nFrames_ * Constants::PI * this->rho0_ / 48.0);
    }
  }
  if (ret.at(0) != 0) {
    ret.at(0) = Constants::GASK_KCAL * this->temperature_ * (ret.at(0) / nwtotal + Constants::EULER_MASC);
    ret.at(1) = Constants::GASK_KCAL * this->temperature_ * (ret.at(1) / nwtotal + Constants::EULER_MASC);
  }
  return ret;
}

/**
 * Calculates the distance between the different atoms in the voxels.
 * Both, for the distance in space, as well as the angular distance.
 * @argument voxel1: The first of the two voxels.
 * @argument n0: The water molecule of the first voxel.
 * @argument voxel2: The voxel to compare voxel1 to.
 * @argument NNd: The lowest distance in space. If the calculated one
 *                is smaller, saves it here.
 * @argument NNs: The lowest distance in angular space. If the calculated
 *                one is smaller, saves it here.
 */
void Action_GIGist::calcTransEntropyDist(int voxel1, int voxel2, int n0, double &NNd, double &NNs) {
  int nSolvV2 = (*this->result_.at(this->dict_.getIndex("population")))[voxel2];
  for (int n1 = 0; n1 < nSolvV2; ++n1) {
    if (voxel1 == voxel2 && n0 == n1) {
      continue;
    }
    double dd = (this->waterCoordinates_.at(voxel1).at(n0) - this->waterCoordinates_.at(voxel2).at(n1)).Magnitude2();
    if (dd > Constants::SMALL && dd < NNd) {
      NNd = dd;
    }
    double rR = this->quaternions_.at(voxel1).at(n0).distance(this->quaternions_.at(voxel2).at(n1));
    double ds = rR * rR + dd;
    if (ds < NNs && ds > 0) {
      NNs = ds;
    }
  }
} 

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
 * Writes a dx file out. The dx file is the same file as the cpptraj dx file, however,
 * this is under my complete control, cpptraj is not.
 * Still for most calculations, the cpptraj tool is used.
 * @argument name: A string holding the name of the written file.
 * @argument data: The data to write to the dx file.
 */
void Action_GIGist::writeDxFile(std::string name, std::vector<double> data) {
  std::ofstream file;
  file.open(name.c_str());
  Vec3 origin = this->center_ - this->dimensions_ * (0.5 * this->voxelSize_);
  file << "object 1 class gridpositions counts " << this->dimensions_[0] << " " << this->dimensions_[1] << " " << this->dimensions_[2] << "\n";
  file << "origin " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
  file << "delta " << this->voxelSize_ << " 0 0\n";
  file << "delta 0 " << this->voxelSize_ << " 0\n";
  file << "delta 0 0 " << this->voxelSize_ << "\n";
  file << "object 2 class gridconnections counts " << this->dimensions_[0] << " " << this->dimensions_[1] << " " << this->dimensions_[2] << "\n";
  file << "object 3 class array type double rank 0 items " << this->nVoxels_ << " data follows" << "\n";
  unsigned int i = 0;
  while ( (i + 3) < this->nVoxels_ ) {
    file << data.at(i) << " " << data.at(i + 1) << " " << data.at(i + 2) << "\n";
    i +=3;
  }
  while (i < this->nVoxels_) {
    file << data.at(i) << " ";
    i++;
  }
  file << std::endl;
}

// Functions used when CUDA is specified.
#ifdef CUDA

/**
 * Frees all the Memory on the GPU.
 */
void Action_GIGist::freeGPUMemory(void) {
  freeCuda(this->NBindex_c_);
  freeCuda(this->molecule_c_);
  freeCuda(this->paramsLJ_c_);
  freeCuda(this->max_c_);
  freeCuda(this->min_c_);
  freeCuda(this->result_w_c_);
  freeCuda(this->result_s_c_);
  freeCuda(this->result_O_c_);
  freeCuda(this->result_N_c_);
  this->NBindex_c_   = NULL;
  this->molecule_c_  = NULL;
  this->paramsLJ_c_  = NULL;
  this->max_c_     = NULL;
  this->min_c_     = NULL;
  this->result_w_c_= NULL;
  this->result_s_c_= NULL;
  this->result_O_c_  = NULL;
  this->result_N_c_  = NULL;
}

/**
 * Copies data from the CPU to the GPU.
 * @throws: CudaException
 */
void Action_GIGist::copyToGPU(void) {
  float grids[3] = {(float)this->gridStart_[0], (float)this->gridStart_[1], (float)this->gridStart_[2]};
  float gride[3] = {(float)this->gridEnd_[0], (float)this->gridEnd_[1], (float)this->gridEnd_[2]};
  try {
    copyMemoryToDevice(&(this->NBIndex_[0]), this->NBindex_c_, this->NBIndex_.size() * sizeof(int));
    copyMemoryToDevice(grids, this->min_c_, 3 * sizeof(float));
    copyMemoryToDevice(gride, this->max_c_, 3 * sizeof(float));
    copyMemoryToDeviceStruct(&(this->charges_[0]), &(this->atomTypes_[0]), this->solvent_, &(this->molecule_[0]), this->numberAtoms_, &(this->molecule_c_),
                              &(this->lJParamsA_[0]), &(this->lJParamsB_[0]), this->lJParamsA_.size(), &(this->paramsLJ_c_));
  } catch (CudaException &ce) {
    this->freeGPUMemory();
    mprinterr("Error: Could not copy data to the device.\n");
    throw ce;
  } catch (std::exception &e) {
    this->freeGPUMemory();
    throw e;
  }
}


#endif
