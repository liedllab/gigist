# GIGIST
GPU Implementation of GIST

If adding something, or adding this to an already existing cpptraj
implementation, you have to update the following files (see below):
+ configure
+ src/Makefile
+ src/CMakeList.txt
+ src/Command.cpp
+ src/cpptrajfiles

Here is what you need to add to the different files:
+ In configure add to the CUDA_TARGET cuda_kernel_gist/lib_cuda_gist.a
+ In the src/Makefile add a recipe for the cuda_kernel_gist/lib_cuda_gist.a, add 
```make
 cuda_kernel_gist/lib_cuda_gist.a:
      cd cuda_kernel_gist/ && $(MAKE) all
```
+ In the src/CMakeLists.txt add at the beginning
  - ```add_subdirectory(cuda_kernel_gist)```
+ In the src/Command.cpp add after the two equivalent GIST statements
  - ```#include "Action_GIGIST.h"```
  - ```Command::AddCmd( new Action_GIGist(),        Cmd::ACT, 1, "gigist" );```
+ In the src/cpptrajfiles add after the equivalent GIST statement
  - ```Action_GIGIST.cpp \```
+ Copy the contents of the directory into your cpptraj src directory


After updating these files, you can simply compile the code via 
make in the root directory.

```bash
$ ./configure gnu -cuda -openmp
$ make
```


The author urges to actually use cuda and openmp or only cuda. Any speedup resulting
from the new implementation will be lost otherwise. The code should still work (at the moment) but will not be fast.



The CUDA source code is its own directory, since it is not officially added in cpptraj yet.
One can easily change that, but needs to also change the commands presented above, as well as
some lines in the include statements.

The columns in the gist output file are ordered differently than in the original implementation, the density
of the water oxygen and hydrogen was moved to the back. This, of course, also has a big impact on the other
columns.





This code was written by Johannes Kraml and is based on the implementation present in cpptraj.

<Johannes.Kraml@uibk.ac.at>


