#!/bin/bash

WD=`pwd`

if [ "$#" -gt 0 ]
then
  if [ "$1" == "-h" ]
	then
		echo "Usage:"
		echo "patch.sh </CPPTRAJ/DIRECTORY>"
		echo "If no argument is given, uses AMBERHOME"
		exit 0
	fi
	CPPTRAJ_HOME="$1"
else
  echo "No argument given, uses AMBERHOME."
	CPPTRAJ_HOME="$AMBERHOME/AmberTools/src/cpptraj/"
fi
echo "cpptraj home set to: $CPPTRAJ_HOME"

cp -r Action_GIGIST.h Action_GIGIST.cpp ExceptionsGIST.h Quaternion.h cuda_kernel_gist/ $CPPTRAJ_HOME/src
cd $CPPTRAJ_HOME && patch -p1 -i $WD/cpptraj.patch

cd $WD
