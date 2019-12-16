#include "mlir/Pass/Pass.h"
#include "mlir/IR/Function.h"
#include "mlir/LoopLifting/access.h"

using namespace mlir;

namespace {
  struct TestAccessMatchers : public FunctionPass<TestAccessMatchers> {
    void runOnFunction() override;
};
} // end anonymous namespace

static inline bool isStore(Operation *op) {
  if (isa<AffineStoreOp>(op)) {
    return true;
  }
  return false;
}

static inline bool isLoad(Operation *op) {
  if (isa<AffineLoadOp>(op)) {
    return true;
  }
  return false;
}

static void collectAccesses(FuncOp f,
    SmallVector<Operation*, 8> &loadOps, SmallVector<Operation*, 8> &storeOps) {
  f.walk([&](Operation *op) {
    if (isLoad(op)) {
      loadOps.push_back(op);
    }
    if (isStore(op)) {
      storeOps.push_back(op);
    }
  });
}

static void matmulABC(FuncOp f) {
  if (f.getNumArguments() != 3) {
    llvm_unreachable("test function expects 3 args");
  }
  SmallVector<Operation*, 8> loadOps;
  SmallVector<Operation*, 8> storeOps;
  collectAccesses(f, loadOps, storeOps);
  
  using namespace looptactics;
  auto _i = placeholder();
  auto _j = placeholder();
  auto _k = placeholder();
  auto _A = arrayPlaceholder();
  auto _B = arrayPlaceholder();
  auto _C = arrayPlaceholder();
  auto psRead = allOf(access(_A, _i, _j), access(_B, _i, _k), access(_C, _k, _j));
  auto psWrite = allOf(access(_A, _i, _j));
  auto matches = match(loadOps, psRead);
  llvm::outs() << "Number of matches: " << matches.size() << "\n";
  matches = match(storeOps, psWrite);
  llvm::outs() << "Number of matches: " << matches.size() << "\n";
}

// Check why it fails.
static void matmulAAC(FuncOp f) {
  if (f.getNumArguments() != 2) {
    llvm_unreachable("test function expects 2 args");
  }
  SmallVector<Operation *, 8> loadOps;
  SmallVector<Operation *, 8> storeOps; 
  collectAccesses(f, loadOps, storeOps);

  using namespace looptactics;
  auto _i = placeholder();
  auto _j = placeholder();
  auto _k = placeholder();
  auto _A = arrayPlaceholder();
  auto _B = arrayPlaceholder();
  auto _C = arrayPlaceholder();
  auto psRead = allOf(access(_A, _i, _j), access(_B, _i, _k), access(_C, _k, _j));
  auto psWrite = allOf(access(_A, _i, _j));
  auto matches = match(loadOps, psRead);
  llvm::outs() << "Number of matches: " << matches.size() << "\n";
  matches = match(storeOps, psWrite);
  llvm::outs() << "Number of matches: " << matches.size() << "\n";
}
  
void TestAccessMatchers::runOnFunction() {
  auto f = getFunction();
  llvm::outs() << f.getName() << "\n";
  if (f.getName() == "matmulABC") {
    matmulABC(f);
  }
  if (f.getName() == "matmulAAC") {
    matmulAAC(f);
  }
}

static PassRegistration<TestAccessMatchers> pass("test-access-matchers",
                                                 "Test C++ access matchers.");
