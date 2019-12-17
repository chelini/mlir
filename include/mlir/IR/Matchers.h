//===- Matchers.h - Various common matchers ---------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matching over MLIR. This mechanism is inspired by LLVM's
// include/llvm/IR/PatternMatch.h.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MATCHERS_H
#define MLIR_MATCHERS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"

namespace mlir {

namespace matchers {

/// Affine pattern.
// FIXME: We have to delay the creation of the AffineExpression 
//as during creation of the placeholder object we don't' know 
//which dimension will index. Therefore we use constant_ and 
//coefficient_ to store the values temporarily. This is not ideal. 
class AffinePattern {
  public:
    MLIRContext *ctx_;
    AffineExpr expr_;
    int64_t constant_;
    int64_t coefficient_;
  
  public:
    AffinePattern() = delete; 
    AffinePattern(MLIRContext *ctx) : ctx_(ctx), expr_(AffineExpr()),
      constant_(0), coefficient_(1) {};
};

/// wrapper around payload (AffinePattern) to bind the
/// placeholder with a given output dimension.
class FixedOutDimPattern {
  public:
    AffinePattern affine_;
    int outDimPos_;
  
  public:
    FixedOutDimPattern() = delete; 
    FixedOutDimPattern(MLIRContext *ctx) :
      affine_(AffinePattern(ctx)), outDimPos_(INT_MAX) {};
};

/// Placeholder.
class Placeholder {
  public:
    FixedOutDimPattern pattern_;
 
  public:
    Placeholder() = delete;
    Placeholder(MLIRContext* ctx) : pattern_(FixedOutDimPattern(ctx)) {};
    void dump() const {
      llvm::outs() << "pattern: " << "\n";
      pattern_.affine_.expr_.dump();
      llvm::outs() << "outDimPos: " << pattern_.outDimPos_ << "\n";
    };
};

inline Placeholder
    operator+(Placeholder p, int64_t i) {
  p.pattern_.affine_.constant_ += i; 
  return p;
}

inline Placeholder
    operator-(Placeholder p, int64_t i) {
  p.pattern_.affine_.constant_ -= i;
  return p;
}

inline Placeholder
    operator*(int64_t i, Placeholder p) {
  if (i <= 0)
    llvm_unreachable("Invalid coefficient for Placeholder");
  p.pattern_.affine_.coefficient_ *= i;
  return p;
}

inline Placeholder
    operator*(Placeholder p, int64_t i) {
  if (i <= 0)
    llvm_unreachable("Invalid coefficient for Placeholder");
  p.pattern_.affine_.coefficient_ *= i;
  return p;
}

} // end namespace matchers

namespace detail {

/// The matcher that matches a certain kind of Attribute and binds the value
/// inside the Attribute.
template <
    typename AttrClass,
    // Require AttrClass to be a derived class from Attribute and get its
    // value type
    typename ValueType =
        typename std::enable_if<std::is_base_of<Attribute, AttrClass>::value,
                                AttrClass>::type::ValueType,
    // Require the ValueType is not void
    typename = typename std::enable_if<!std::is_void<ValueType>::value>::type>
struct attr_value_binder {
  ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  attr_value_binder(ValueType *bv) : bind_value(bv) {}

  bool match(const Attribute &attr) {
    if (auto intAttr = attr.dyn_cast<AttrClass>()) {
      *bind_value = intAttr.getValue();
      return true;
    }
    return false;
  }
};

/// The matcher that matches a constant foldable operation that has no side
/// effect, no operands and produces a single result.
template <typename AttrT> struct constant_op_binder {
  AttrT *bind_value;

  /// Creates a matcher instance that binds the constant attribute value to
  /// bind_value if match succeeds.
  constant_op_binder(AttrT *bind_value) : bind_value(bind_value) {}

  bool match(Operation *op) {
    if (op->getNumOperands() > 0 || op->getNumResults() != 1)
      return false;
    if (!op->hasNoSideEffect())
      return false;

    SmallVector<OpFoldResult, 1> foldedOp;
    if (succeeded(op->fold(/*operands=*/llvm::None, foldedOp))) {
      if (auto attr = foldedOp.front().dyn_cast<Attribute>()) {
        if ((*bind_value = attr.dyn_cast<AttrT>()))
          return true;
      }
    }
    return false;
  }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// integer operation and binds the constant integer value.
struct constant_int_op_binder {
  IntegerAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_int_op_binder(IntegerAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0)->getType();

    if (type.isIntOrIndex()) {
      return attr_value_binder<IntegerAttr>(bind_value).match(attr);
    }
    if (type.isa<VectorType>() || type.isa<RankedTensorType>()) {
      if (auto splatAttr = attr.dyn_cast<SplatElementsAttr>()) {
        return attr_value_binder<IntegerAttr>(bind_value)
            .match(splatAttr.getSplatValue());
      }
    }
    return false;
  }
};

/// The matcher that matches a given target constant scalar / vector splat /
/// tensor splat integer value.
template <int64_t TargetValue> struct constant_int_value_matcher {
  bool match(Operation *op) {
    APInt value;
    return constant_int_op_binder(&value).match(op) && TargetValue == value;
  }
};

/// The matcher that matches anything except the given target constant scalar /
/// vector splat / tensor splat integer value.
template <int64_t TargetNotValue> struct constant_int_not_value_matcher {
  bool match(Operation *op) {
    APInt value;
    return constant_int_op_binder(&value).match(op) && TargetNotValue != value;
  }
};

/// The matcher that matches a certain kind of op.
template <typename OpClass> struct op_matcher {
  bool match(Operation *op) { return isa<OpClass>(op); }
};

/// The matcher that matches Affine/Store ops with a given pattern expressed
/// via Placeholder(...)
// FIXME: we are mixing namespace here (use of matchers in detail).
// Maybe we can also live without "outDimPos_".
template <typename OpClass> struct op_load_store_matcher {
  
  std::vector<matchers::Placeholder> placeholders_;
  
  op_load_store_matcher(std::vector<matchers::Placeholder> ps) : placeholders_(ps) {
    int pos = 0;
    for (auto &placeholder : placeholders_) { 
      placeholder.pattern_.outDimPos_ = pos;
      // At this point we know the placeholder postion.
      // We create the affine expression to match.
      detail::bindDims(
        placeholder.pattern_.affine_.ctx_, 
        placeholder.pattern_.affine_.expr_, pos++);
      placeholder.pattern_.affine_.expr_ = placeholder.pattern_.affine_.expr_ + 
        placeholder.pattern_.affine_.constant_;
      placeholder.pattern_.affine_.expr_ = placeholder.pattern_.affine_.expr_ *
        placeholder.pattern_.affine_.coefficient_;
    }
  };
  op_load_store_matcher() = delete;
  bool match(Operation *op) { 
    if (!placeholders_.size()) {
      llvm_unreachable("expect non empty placeholders");
    }
    if (!op) {
      return false;
    }
    if (!isa<OpClass>(op)) {
      return false;
    }
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      size_t dims = loadOp.getAffineMap().getNumResults();
      if (dims != placeholders_.size()) {
        return false;
      }
      for (size_t dim = 0; dim < dims; dim++) {
        AffineExpr loadAffine = loadOp.getAffineMap().getResult(dim);
        loadAffine.dump();
        if (placeholders_[dim].pattern_.affine_.expr_ != loadAffine) {
          return false;
        }
      }
      return true;
    }
    if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      size_t dims = storeOp.getAffineMap().getNumResults();
      if (dims != placeholders_.size()) {
        return false;
      }
      for (size_t dim = 0; dim < dims; dim++) {
        AffineExpr storeAffine = storeOp.getAffineMap().getResult(dim);
        storeAffine.dump();
        if (placeholders_[dim].pattern_.affine_.expr_ != storeAffine) {
          return false; 
        }
      }
      return true;
    }
    llvm_unreachable("expect AffineStore or AffineLoad");
  };
};

/// Trait to check whether T provides a 'match' method with type
/// `OperationOrValue`.
template <typename T, typename OperationOrValue>
using has_operation_or_value_matcher_t =
    decltype(std::declval<T>().match(std::declval<OperationOrValue>()));

/// Statically switch to a Value matcher.
template <typename MatcherClass>
typename std::enable_if_t<is_detected<detail::has_operation_or_value_matcher_t,
                                      MatcherClass, Value *>::value,
                          bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  return matcher.match(op->getOperand(idx));
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
typename std::enable_if_t<is_detected<detail::has_operation_or_value_matcher_t,
                                      MatcherClass, Operation *>::value,
                          bool>
matchOperandOrValueAtIndex(Operation *op, unsigned idx, MatcherClass &matcher) {
  if (auto defOp = op->getOperand(idx)->getDefiningOp())
    return matcher.match(defOp);
  return false;
}

/// Terminal matcher, always returns true.
struct AnyValueMatcher {
  bool match(Value *op) const { return true; }
};

/// Binds to a specific value and matches it.
struct PatternMatcherValue {
  PatternMatcherValue(Value *val) : value(val) {}
  bool match(Value *val) const { return val == value; }
  Value *value;
};

template <typename TupleT, class CallbackT, std::size_t... Is>
constexpr void enumerateImpl(TupleT &&tuple, CallbackT &&callback,
                             std::index_sequence<Is...>) {
  (void)std::initializer_list<int>{
      0,
      (callback(std::integral_constant<std::size_t, Is>{}, std::get<Is>(tuple)),
       0)...};
}

template <typename... Tys, typename CallbackT>
constexpr void enumerate(std::tuple<Tys...> &tuple, CallbackT &&callback) {
  detail::enumerateImpl(tuple, std::forward<CallbackT>(callback),
                        std::make_index_sequence<sizeof...(Tys)>{});
}

/// RecursivePatternMatcher that composes.
template <typename OpType, typename... OperandMatchers>
struct RecursivePatternMatcher {
  RecursivePatternMatcher(OperandMatchers... matchers)
      : operandMatchers(matchers...) {}
  bool match(Operation *op) {
    if (!isa<OpType>(op) || op->getNumOperands() != sizeof...(OperandMatchers))
      return false;
    bool res = true;
    enumerate(operandMatchers, [&](size_t index, auto &matcher) {
      res &= matchOperandOrValueAtIndex(op, index, matcher);
    });
    return res;
  }
  std::tuple<OperandMatchers...> operandMatchers;
};

} // end namespace detail

/// Matches a value from a constant foldable operation and writes the value to
/// bind_value.
template <typename AttrT>
inline detail::constant_op_binder<AttrT> m_Constant(AttrT *bind_value) {
  return detail::constant_op_binder<AttrT>(bind_value);
}

/// Matches a constant scalar / vector splat / tensor splat integer one.
inline detail::constant_int_value_matcher<1> m_One() {
  return detail::constant_int_value_matcher<1>();
}

/// Matches the given OpClass.
template <typename OpClass> inline detail::op_matcher<OpClass> m_Op() {
  return detail::op_matcher<OpClass>();
}

template<class T, class...>
struct are_same : std::true_type
{};

template<class T, class U, class... TT>
struct are_same<T, U, TT...>
    : std::integral_constant<bool, std::is_same<T,U>{} && are_same<T, TT...>{}>
{};

template <typename OpClass, typename... Args> 
    inline detail::op_load_store_matcher<OpClass> m_Op(matchers::Placeholder arg, Args... args) {
  static_assert(are_same<matchers::Placeholder, Args...>{},
    "all args must be Placeholder");
  return detail::op_load_store_matcher<OpClass>({arg, args...});
}

/// Matches a constant scalar / vector splat / tensor splat integer zero.
inline detail::constant_int_value_matcher<0> m_Zero() {
  return detail::constant_int_value_matcher<0>();
}

/// Matches a constant scalar / vector splat / tensor splat integer that is any
/// non-zero value.
inline detail::constant_int_not_value_matcher<0> m_NonZero() {
  return detail::constant_int_not_value_matcher<0>();
}

/// Entry point for matching a pattern over a Value.
template <typename Pattern>
inline bool matchPattern(Value *value, const Pattern &pattern) {
  // TODO: handle other cases
  if (auto *op = value->getDefiningOp())
    return const_cast<Pattern &>(pattern).match(op);
  return false;
}

/// Entry point for matching a pattern over an Operation.
template <typename Pattern>
inline bool matchPattern(Operation *op, const Pattern &pattern) {
  return const_cast<Pattern &>(pattern).match(op);
}

/// Matches a constant holding a scalar/vector/tensor integer (splat) and
/// writes the integer value to bind_value.
inline detail::constant_int_op_binder
m_ConstantInt(IntegerAttr::ValueType *bind_value) {
  return detail::constant_int_op_binder(bind_value);
}

template <typename OpType, typename... Matchers>
auto m_Op(Matchers... matchers) {
  return detail::RecursivePatternMatcher<OpType, Matchers...>(matchers...);
}

namespace matchers {
inline auto m_Any() { return detail::AnyValueMatcher(); }
inline auto m_Val(Value *v) { return detail::PatternMatcherValue(v); }
} // namespace matchers

} // end namespace mlir

#endif // MLIR_MATCHERS_H
