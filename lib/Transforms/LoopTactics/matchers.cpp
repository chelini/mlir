#include "matchers.h"

namespace matchers {

using namespace mlir;

OperationType toMatcherType(Operation *op) {
  if (isa<AffineForOp>(op))
    return OperationType::AffineFor;
  else return OperationType::Unknown;
}

bool OperationMatcher::isMatching(const OperationMatcher &matcher,
    Operation *op) {
  if (!op)
    return false;
  if (toMatcherType(op) != matcher.current_)
    return false; 
  if (matcher.callback_ && !matcher.callback_(op))
    return false;
  size_t nChildren = matcher.children_.size();
  for (size_t i = 0; i < nChildren; i++) {
    // The only matcher allowed to have a child is affineFor
    // which represents an AffineForOp. We also know that
    // an AffineForOp has a single region which contains a single
    // block.
    if (!isMatching(matcher.children_.at(i), &op->getRegion(0).front().front())) {
      return false;
    }
  }
  return true;
}

} // end namespace matchers
