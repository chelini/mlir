#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::edsc;

namespace {
  struct LiftingAffineToLinalg : public FunctionPass<LiftingAffineToLinalg> {
    void runOnFunction() override;
  };
}

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

void LiftingAffineToLinalg::runOnFunction() {
  auto f = getFunction();
  SmallVector<Operation *, 8> loadOps;
  SmallVector<Operation *, 8> storeOps;
  collectAccesses(f, loadOps, storeOps);
    
  if (!storeOps.size())
    return;  

  using namespace matchers;
  auto _i = Placeholder(f.getContext());
  auto _j = Placeholder(f.getContext());
  auto res = matchPattern(storeOps[0], m_Op<AffineStoreOp>(3 * _i + 1, _j));
  llvm::outs() << "matched ? " << res << "\n";
  res = matchPattern(storeOps[0], m_Op<AffineStoreOp>(_i, _j));
  llvm::outs() << "matched ? " << res << "\n";
  res = matchPattern(storeOps[0], m_Op<AffineStoreOp>(_i));
  llvm::outs() << "matched ? " << res << "\n";
}

static PassRegistration<LiftingAffineToLinalg>
  LiftingAffineToLinalg("lift-affine-to-linalg", "lift affine to linalg");
