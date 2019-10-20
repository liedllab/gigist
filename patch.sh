#!/bin/bash

WD=`pwd`

if [ "$#" -gt 0 ]
then
	CPPTRAJ_HOME="$1"
else
	CPPTRAJ_HOME="$AMBERHOME/AmberTools/src/cpptraj/"
fi
echo "cpptraj home set to: $CPPTRAJ_HOME"

cp -r Action_GIGIST.h Action_GIGIST.cpp ExceptionsGIST.h Quaternion.h cuda_kernel_gist/ $CPPTRAJ_HOME/src
cd $CPPTRAJ_HOME && patch -p1 -i $WD/cpptraj.patch

cd $WD
