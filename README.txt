################################################################################
Current dependencies:
- src/main and src/test:
  - Java 5+, but EJML class files are 1.6
  - EJML-core-0.29.jar (https://github.com/lessthanoptimal/ejml)
  - EJML-dense64-0.29.jar
################################################################################
EJML mult sparse, 1.0, 2016/01/28

Provides a fast multiplication for sparse (i.e. with mostly zeros)
RowD1Matrix64F matrices, which are "dense" in that they are backed by a dense
array, but can still greatly benefit from sparsity related algorithms.

################################################################################
