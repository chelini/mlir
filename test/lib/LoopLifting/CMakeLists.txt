add_llvm_library(MLIRTestLoopLifting
  TestAccessMatchers.cpp
  )

target_link_libraries(MLIRTestLoopLifting
  MLIRPass
  )
