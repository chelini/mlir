#ifndef ACCESS_H
#define ACCESS_H

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"

using namespace mlir;
using namespace llvm;

namespace looptactics {

class CandidateDim;

/// Simple pattern for placeholder which
/// captures and affine expr.
class AffineFunction {
  public:
    int coefficient_;
    int constant_;

  public:
    AffineFunction() : coefficient_(1), constant_(0) {};
    static std::vector<CandidateDim>
      candidates(Operation *op, const AffineFunction &pattern, int outDimPos);
};

/// Candidate.
class CandidateDim {
  public:
    mlir::Value *inputDimPos_;
    Operation *operation_;
 
  public:
    CandidateDim() : inputDimPos_(nullptr), operation_(nullptr) {};
    CandidateDim(mlir::Value *v, Operation *o) : inputDimPos_(v), operation_(o) {};
    static std::vector<CandidateDim>
        candidates(Operation *op, const AffineFunction &pattern);
    bool operator==(const CandidateDim &candidate) const;
    bool operator!=(const CandidateDim &candidate) const;
};

/// Fixed output dim pattern.
/// Wrapper around the payload (i.e., AffineFunction) to keep track
/// of the output dimensions the placeholder refers to.
class FixedOutDimPattern {
  public:
    AffineFunction affine_;
    int outDimPos_;

  public:
    explicit FixedOutDimPattern(AffineFunction f, int pos) :
      affine_(f), outDimPos_(pos) {};
    explicit FixedOutDimPattern(int pos) :
      affine_(AffineFunction()), outDimPos_(pos) {};
    FixedOutDimPattern() : affine_(AffineFunction()), outDimPos_(INT_MAX) {};
    static std::vector<CandidateDim>
      candidates(Operation *op, const FixedOutDimPattern &pattern);
};

/// Placeholder class.
template <typename PatternPayload>
class Placeholder {
  public:
    PatternPayload pattern_;
    std::vector<CandidateDim> candidates_;
    const size_t id_;

  public:
    explicit Placeholder() : 
      pattern_(FixedOutDimPattern()), candidates_({}), id_(nextId_++) {};
    explicit Placeholder(PatternPayload pattern) : 
      pattern_(pattern), candidates_({}), id_(nextId_++) {}; 
    void dump();
  
  private:
    static thread_local size_t nextId_;
};

inline Placeholder<FixedOutDimPattern> 
    operator+(Placeholder<FixedOutDimPattern> p, int i) {
  p.pattern_.affine_.constant_ += i;
  return p;
}

inline Placeholder<FixedOutDimPattern> 
    operator-(Placeholder<FixedOutDimPattern> p, int i) {
  p.pattern_.affine_.constant_ -= i;
  return p;
}

inline Placeholder<FixedOutDimPattern> 
    operator*(int i, Placeholder<FixedOutDimPattern> p) {
  if (i <= 0)
    llvm_unreachable("Invalid coefficient for Placeholder");
  p.pattern_.affine_.coefficient_ *= i;
  return p;
}

inline Placeholder<FixedOutDimPattern> 
    operator*(Placeholder<FixedOutDimPattern> p, int i) {
  if (i <= 0)
    llvm_unreachable("Invalid coefficient for Placeholder");
  p.pattern_.affine_.coefficient_ *= i;
  return p;
}

/// Object returned by call of allOf.
/// single access can be constructed using access(...)
class PlaceholderSet {
  public:
    std::vector<Placeholder<FixedOutDimPattern>> placeholders_;
    // Each inner vector has a set of indices of placeholders that should appear
    // together in a relation.  Different groups must correspond to different
    // relations.  We store indices separately because a placeholder may appear
    // in multiple relations, actual objects are stored in placeholders_.  We
    // don't store references because of how the match is currently structured: a
    // vector of candidates, each of which is itself a vector with the same index
    // as the position of the placeholder in placeholders_.  This may change in
    // the future for a more C++-idiomatic API.
    std::vector<std::vector<size_t>> placeholderGroups_;
    // Placeholder fold is an identifier of a set of placeholders that must get
    // assigned the same candidate value modulo the matched memRef.  The idea is to
    // reuse, at the API level, placeholders in multiple places to indicate
    // equality of the matched access patterns.
    // This vector is co-indexed with placeholders_.  By default, each
    // placeholder gets assigned its index in the placeholders_ list, that is
    // placeholderFolds_[i] == i. Placeholders that belong to the same group have
    // the same fold index, by convention we assume it is the index in
    // placeholders_ of the first placeholder in the fold.
    // One placeholder cannot belong to multiple folds.
    std::vector<size_t> placeholderFolds_;

  public:
    explicit PlaceholderSet() = default;
    void dump();
    void setUpFolds();
    decltype (placeholders_.begin()) begin() {
      return placeholders_.begin();
    };
    decltype (placeholders_.cbegin()) begin() const {
      return placeholders_.cbegin();
    };
    decltype (placeholders_.end()) end() {
      return placeholders_.end();
    };
    decltype (placeholders_.cend()) end() const {
      return placeholders_.cend();
    }; 
};

// forward declaration.
class MatchCandidate;
// A match. Bind the placeholder id with a candidate dimension.
class Match {
  public:
    explicit Match(const PlaceholderSet &ps, const std::vector<CandidateDim> &candidates);
    MatchCandidate operator[](const Placeholder<FixedOutDimPattern> &placeholder) const;

    std::vector<std::pair<size_t, CandidateDim>> placeholderValues_;
};
using Matches = std::vector<Match>;

/// Matched candidate.
class MatchCandidate {
  public:
    MatchCandidate() : assigned_(false) {};
    mlir::Value* matchedDim() const {
      return candidateDimension_.inputDimPos_;
    }
  private:
    bool assigned_;
    CandidateDim candidateDimension_;
  friend class Match;
};

Matches match (const SmallVector<Operation *, 8> &opsFirst, PlaceholderSet psFirst,
    const SmallVector<Operation *, 8> &opsSecond = SmallVector<Operation *, 8>(), 
    PlaceholderSet psSecond = PlaceholderSet());

using placeholder = Placeholder<FixedOutDimPattern>;

} // end namespace looptactics.

#include "access-inl.h"

#endif
