add_llvm_library(MLIRLoopTactics
  matchers.cpp
  access.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LoopLifting
  ) 
