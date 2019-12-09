#ifndef MATCHER_H
#define MATCHER_H

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"

namespace matchers {

using namespace mlir;

class OperationMatcher;

OperationMatcher affineFor(std::function<bool(Operation *op)> callback,
    OperationMatcher &&);
OperationMatcher affineFor(std::function<bool(Operation *op)> callback);
OperationMatcher affineFor(OperationMatcher &&);

enum class OperationType {
  AffineFor,
  Unknown
};

class OperationMatcher {
  friend OperationMatcher affineFor(std::function<bool(Operation *op)> callback,
      OperationMatcher &&child);
  friend OperationMatcher affineFor(OperationMatcher &&child);
  friend OperationMatcher affineFor(std::function<bool(Operation *op)> callback);

  public:
    static bool isMatching(const OperationMatcher &matcher, Operation *op);
  private:
    OperationType current_;
    std::vector<OperationMatcher> children_;
    std::function<bool(Operation *op)> callback_;
};


} // end namespace matchers

#include "matchers-inl.h"

#endif // MATCHER_H
