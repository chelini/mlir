#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "LoopTactics/matchers.h"
#include "LoopTactics/access.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/StandardOps/Ops.h"

// check me.
#include "mlir/Dialect/LoopOps/LoopOps.h"

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "lift-loops"

/* Thinking:

- Indeally a given structural matcher of type loop
is able to capture different loop types from different
dialect (i.e., AffinForOp or loop::ForOp. See if we can
do something similar to the operation matchers in matchers.h.
We also need to make sure that isMatching works if different
for loop types are provided. For example, AffineForOp has a single
region with a single block. Is this the case also for others type 
of loops (i.e., loop::FrOp)? 

*/

namespace {
  class LoopsLifter : public FunctionPass<LoopsLifter> {
    public:
      explicit LoopsLifter() {};
      void runOnFunction() override;
  };
}

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLoopsLiftingPass() {
  return std::make_unique<LoopsLifter>();
}

void LoopsLifter::runOnFunction() {

  using namespace looptactics;
  auto matcher = 
    m_Loop<loop::ForOp>(
      m_Loop<loop::ForOp>(
        m_Loop<loop::ForOp>(hasGemmPattern<loop::ForOp>)));

  auto inspectBlock = [&](Operation *op) {
    if (isMatching<loop::ForOp>(matcher, op)) {
      LLVM_DEBUG(dbgs() << "Pattern is matched!\n");
    }
  };
  
  for (auto &block: getFunction()) {
    block.walk(inspectBlock);
  }

} 

static PassRegistration<LoopsLifter> pass("lift-loops", "lift loops with loop tactics");
