// RUN: mlir-opt %s -disable-pass-threading=true -test-access-matchers -o /dev/null 2>&1 | FileCheck %s

func @matmulABC(%A: memref<64x64xf64>, %B: memref<64x64xf64>, %C: memref<64x64xf64>) {
  affine.for %arg3 = 0 to 64 {
  affine.for %arg4 = 0 to 64 {
   affine.for %arg64 = 0 to 64 {
    %a = affine.load %A[%arg3, %arg64] : memref<64x64xf64>
    %b = affine.load %B[%arg64, %arg4] : memref<64x64xf64>
    %ci = affine.load %C[%arg3, %arg4] : memref<64x64xf64>
    %p = mulf %a, %b : f64
    %co = addf %ci, %p : f64
    affine.store %co, %C[%arg3, %arg4] : memref<64x64xf64>
   }
  }
 }
 return
}

// CHECK-LABEL: matmulABC
//       CHECK: Number of matches: 1
//       CHECK: Number of matches: 1

func @matmulAAC(%A: memref<64x64xf64>, %C: memref<64x64xf64>) {
  affine.for %arg3 = 0 to 64 {
  affine.for %arg4 = 0 to 64 {
   affine.for %arg64 = 0 to 64 {
    %a = affine.load %A[%arg3, %arg64] : memref<64x64xf64>
    %b = affine.load %A[%arg64, %arg4] : memref<64x64xf64>
    %ci = affine.load %C[%arg3, %arg4] : memref<64x64xf64>
    %p = mulf %a, %b : f64
    %co = addf %ci, %p : f64
    affine.store %co, %C[%arg3, %arg4] : memref<64x64xf64>
   }
  }
 }
 return
}

// CHECK-LABEL: matmulAAC
//       CHECK: Number of matches: 0
//       CHECK: Number of matches: 1
