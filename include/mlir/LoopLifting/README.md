# Loop Tactics

Original code developed by Alex (https://ozinenko.com/) and availble at https://github.com/PollyLabs/islutils

Publications: https://hal.inria.fr/hal-01965599/document

```
// C += A * B.
func @matmul(%A: memref<64x32xf64>, %B: memref<32x64xf64>, %C: memref<64x64xf64>) {
 affine.for %arg3 = 0 to 64 {
  affine.for %arg4 = 0 to 64 {
   affine.for %arg64 = 0 to 64 {
    %a = affine.load %A[%arg3, %arg64] : memref<64x32xf64>
    %b = affine.load %B[%arg64, %arg4] : memref<32x64xf64>
    %ci = affine.load %C[%arg3, %arg4] : memref<64x64xf64>
    %p = mulf %a, %b : f64
    %co = addf %ci, %p : f64
    affine.store %co, %C[%arg3, %arg4] : memref<64x64xf64>
   }
  }
 }
 return
}
func @main() {
 %A = alloc() : memref<64x32xf64>
 %B = alloc() : memref<32x64xf64>
 %C = alloc() : memref<64x64xf64>
 %cf1 = constant 1.00000e+00 : f64
 linalg.fill(%A, %cf1) : memref<64x32xf64>, f64
 linalg.fill(%B, %cf1) : memref<32x64xf64>, f64
 linalg.fill(%C, %cf1) : memref<64x64xf64>, f64
 call @matmul(%A, %B, %C) : (memref<64x32xf64>, memref<32x64xf64>, memref<64x64xf64>) -> ()
 call @print_memref_2d_f64(%C): (memref<64x64xf64>) -> ()
 return
}
func @print_memref_2d_f64(memref<64x64xf64>)
```

```
mlir-opt gemm.mlir -loop-tactics -debug-only=loop-tactics
``` 
