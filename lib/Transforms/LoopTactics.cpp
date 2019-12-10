#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "LoopTactics/matchers.h"
#include "LoopTactics/access.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/StandardOps/Ops.h"

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "loop-tactics"

namespace {
  class LoopTactics : public FunctionPass<LoopTactics> {
    public:
      explicit LoopTactics() {};
      void runOnFunction() override;
  };
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLoopTacticsPass() {
  return std::make_unique<LoopTactics>();
}

//static void inspectForOp(AffineForOp forOp) {
//  // Assuming a perfectly nested loop 
//  // jump to innermost loop.
//  while (1) {
//    Block &body = forOp.region().front();
//    if (body.begin() != std::prev(body.end(), 2))
//      break;
//    forOp = dyn_cast<AffineForOp>(&body.front());
//  }
//  SmallVector<Operation *, 8> loadOps;
//  forOp.getOperation()->walk([&](Operation *op) {
//    if (isa<AffineLoadOp>(op))
//      loadOps.push_back(op);
//  });
//
//  using namespace matchers;  
//  auto _i = Placeholder();
//  auto _j = Placeholder();
//  auto _k = Placeholder();
//  auto ps = allOf(access(_i, _j), access(_i, _k), access(_k, _j));
//  match(loadOps, ps);
//}

//void LoopTactics::runOnFunction() {
//  for (auto &block : getFunction())  
//    for (auto &op : block) 
//      if (auto forOp = dyn_cast<AffineForOp>(op)) 
//        inspectForOp(forOp);
//}

static bool hasGemmAccesses(SmallVector<Operation *, 8> loadOps) {
  using namespace looptactics;
  auto _i = Placeholder();
  auto _j = Placeholder();
  auto _k = Placeholder();
  auto psRead = allOf(access(_i, _j), access(_i, _k), access(_k, _j));
  auto matches = match(loadOps, psRead);
  return matches.size() == 1;
}

static bool hasGemmOperations(Operation *op) { 
  using namespace matchers;
  auto matcher = m_Op<AddFOp>(m_Any(), m_Op<MulFOp>());
  //auto matcher = m_Op<AddFOp>(m_Op<MulFOp>(), m_any());
  auto forOp = dyn_cast<AffineForOp>(op); 
  int count = 0;
  forOp.getOperation()->walk([&] (Operation *op) {
    if (matcher.match(op))
      count++;
  });

  return count == 1;
}

static bool hasGemmPatternImpl(Operation *op) {
  SmallVector<Operation *, 8> loadOps;
  auto forOp = dyn_cast<AffineForOp>(op);
  
  forOp.getOperation()->walk([&] (Operation *op) {
    if (isa<AffineLoadOp>(op))
      loadOps.push_back(op);
  });
  
  // expect 3 reads.
  if (loadOps.size() != 3)  
    return false;
  return hasGemmOperations(op) && hasGemmAccesses(loadOps);
}

void LoopTactics::runOnFunction() {

  auto hasGemmPattern = [&](Operation *op) {
    return hasGemmPatternImpl(op);
  };

  using namespace looptactics;
  auto matcher = 
    affineFor(
      affineFor(
        affineFor(hasGemmPattern)));
  
  for (auto &block: getFunction()) {
    for (auto &op : block) {
      if (OperationMatcher::isMatching(matcher, &op)) {
        LLVM_DEBUG(dbgs() << "Gemm matched!\n");
      } else {
        LLVM_DEBUG(dbgs() << "Gemm not matched!\n");  
      }
    }
  }
} 

static PassRegistration<LoopTactics> pass("loop-tactics", "enable loop tactics");
