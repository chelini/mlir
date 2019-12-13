#ifndef MATCHER_H
#define MATCHER_H

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace looptactics {

using namespace mlir;
using namespace llvm;

template <typename OpType>
struct Matcher {

  Matcher<OpType> m_Loop(std::function<bool(OpType)>);
  Matcher<OpType> m_Loop(Matcher<OpType> &&);
  
  bool hasSameType(Operation *op) const { return isa<OpType>(op); };
  std::vector<Matcher<OpType>> children_;
  std::function<bool(OpType op)> callback_;
};

} // end namespace looptactics.

#include "matchers-inl.h"

#endif // MATCHER_H
