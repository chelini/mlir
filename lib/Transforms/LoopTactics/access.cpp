#include "access.h"
#include "llvm/Support/raw_ostream.h"

// TODO: use SmallVector instead of std::vector.
// TODO: bring matching for array name and capture the
// matched array name. This is really useful if we want
// to transparently call BLAS functions.
// placeholder are not reused among different calls of allOf
// fix this or let the match method ensure that the same placeholder
// used for read and write matches the same dimension. 

namespace looptactics {

// Is the affine expression single-dimension?
static bool isSingleDim(AffineExpr expr) {
  int dimCount = 0;
  expr.walk([&dimCount](AffineExpr expr) {
    if (expr.getKind() == AffineExprKind::DimId)
      dimCount++;
    });
  return dimCount == 1;
}

// Check if the affine expression bounded to "op" at position
// "outDimPos" satisfies the pattern "pattern".
std::vector<CandidateDim> AffineFunction::candidates
    (Operation *op, const AffineFunction &pattern, int outDimPos) {
  if (outDimPos == INT_MAX) {
    llvm_unreachable("Invalid outDimPos dimension");
  }

  if ((!isa<AffineLoadOp>(op)) && (!isa<AffineStoreOp>(op))) {
    llvm_unreachable("Expect only (affine) load and store at this point");
  }

  // TODO: here we assume an AffineLoadOp, but we may want
  // to extend to generic load/store ops. 
  std::vector<CandidateDim> results{};  
  AffineExpr affineExpr;
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    affineExpr = loadOp.getAffineMap().getResult(outDimPos);
  }
  if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    affineExpr = storeOp.getAffineMap().getResult(outDimPos);
  }

  // bail out if the affine expression is not
  // a single dimension affine map.
  if (!isSingleDim(affineExpr)) {
    return {};
  }

  struct AffineBody {
    int64_t coefficient = 1;
    int64_t constant = 0;
  }body;

  affineExpr.walk([&](AffineExpr expr) { 
    if ((expr.getKind() == AffineExprKind::Mul) || (expr.getKind() == AffineExprKind::Add)) {
      auto rhs = expr.dyn_cast<AffineBinaryOpExpr>().getRHS();
      if (rhs.getKind() == AffineExprKind::Constant) {
        if (expr.getKind() == AffineExprKind::Mul) {
          body.coefficient = rhs.dyn_cast<AffineConstantExpr>().getValue();
        }
        else {
          body.constant = rhs.dyn_cast<AffineConstantExpr>().getValue();
        }
      }
    }
  });

  if (pattern.coefficient_ != body.coefficient) {
    return {};
  }
  if (pattern.constant_ != body.constant) {
    return {};
  }
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    SmallVector<Value*, 2> operands(loadOp.getMapOperands());
    results.push_back(CandidateDim{operands[outDimPos], op});
  }
  if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    SmallVector<Value*, 2> operands(storeOp.getMapOperands());
    results.push_back(CandidateDim{operands[outDimPos], op});
  }
  return results;
}

std::vector<CandidateDim>
    FixedOutDimPattern::candidates(Operation *op, const FixedOutDimPattern &pattern) {
  return AffineFunction::candidates(op, pattern.affine_, pattern.outDimPos_);
}

// Check that, if two elements in "combination" correspond to the same values
// in "folds", they are equal, and that they are unique within "combination"
// otherwise.  Comparison is performed by calling the function objects
// "eqCompare" and "neCompare" for equality and non-equality respectively.
// While these operations are often reciprocal, this is not always the case,
// for example in tri-state logic.
// "folds" must be at least as large as "combination".
template <typename T, typename EqComparator, typename NeComparator>
bool areFoldsValid(const std::vector<T> &combination,
                   const std::vector<size_t> &folds, EqComparator eqCompare,
                   NeComparator neCompare) {
  // Algorithmically not the most efficient way of finding duplicates, but
  // removes the need to include hash-tables and/or perform additional
  // allocations.
  size_t size = combination.size();
  if (size > folds.size()) {
    llvm_unreachable("folds are not properly set up");
  }

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      if (folds[i] == folds[j]) {
        if (neCompare(combination.at(i), combination.at(j))) {
          return false;
        } else {
          continue;
        }
      }
      if (eqCompare(combination.at(i), combination.at(j))) {
        return false;
      }
    }
  }
  return true;
}

// All the placeholders should get different assignments, expect those
// that belong to the same fold.
static bool hasNoDuplicateAssignments(
    const PlaceholderSet &ps, const std::vector<CandidateDim> &partialCombination) {
  return areFoldsValid(partialCombination, ps.placeholderFolds_,
    [](const CandidateDim left, const CandidateDim right) {
      return left == right;
    },
    [](const CandidateDim left, const CandidateDim right) {
      return left != right;
    });
}

// All the placeholders in a group are either not yet matched, or
// matched the same load/store operation. A load/store operation 
// matched in a group is not matched in any previous group.
static bool groupsAreProperlyFormed(
    const PlaceholderSet &ps, const std::vector<CandidateDim> &partialCombination) {
  std::vector<Operation *> previouslyMatchedOperations{};

  for (const auto &group : ps.placeholderGroups_) {
    Operation *matchedOp = nullptr;
    for (size_t pos : group) {
      if (pos >= partialCombination.size()) {
        continue;
      }
      Operation *candidateOp = partialCombination.at(pos).operation_;
      if (matchedOp) {
        if (matchedOp != candidateOp) {
          return false;
        }
      } else {
        matchedOp = candidateOp;
        auto it = std::find(previouslyMatchedOperations.begin(),
                            previouslyMatchedOperations.end(), matchedOp);
        if (it != previouslyMatchedOperations.end()) {
          return false;
        }
        previouslyMatchedOperations.push_back(matchedOp);
      }
    }
  }
  return true;
}

bool PlaceholderSet::isSuitableCombination(
    const std::vector<CandidateDim> &partialCombination) const {
  return hasNoDuplicateAssignments(*this, partialCombination) &&
         groupsAreProperlyFormed(*this, partialCombination);
}

static inline Operation *findOp(const std::vector<size_t> &group,
    const std::vector<CandidateDim> &combination) {
  for (auto idx : group) {
    if (idx >= combination.size()) {
      continue;
    }
    return combination.at(idx).operation_;
  }
  return nullptr;
}

Value* getMemRef(Operation *op) {
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    return loadOp.getMemRef();
  }
  else {
    if (!isa<AffineStoreOp>(op))
      llvm_unreachable("expect load/store");
    auto storeOp = dyn_cast<AffineStoreOp>(op);
    return storeOp.getMemRef();
  }
}

static bool compareGroupsBelongToSameArray(
    const std::vector<size_t> &group1, const std::vector<size_t> &group2,
    const std::vector<CandidateDim> &combination, bool equality) {
  Operation *operation1 = findOp(group1, combination);
  Operation *operation2 = findOp(group2, combination);
  if ((!operation1) || (!operation2))
    return false;

  // TODO: generalize for generic load/store operations. 
  if ((!isa<AffineLoadOp>(operation1)) && (!isa<AffineStoreOp>(operation2)))
    llvm_unreachable("expect load or store operation");
 
  auto memRefOp1 = getMemRef(operation1);
  auto memRefOp2 = getMemRef(operation2);  
  return (memRefOp1 == memRefOp2) ^ !equality;
}

bool PlaceholderGroupedSet::isSuitableCombination(
    const std::vector<CandidateDim> &partialCombination) const {
  bool isValid = 
    static_cast<const PlaceholderSet&>(*this).isSuitableCombination(partialCombination);
  isValid &&
    areFoldsValid(this->placeholderGroups_, placeholderGroupFolds_,
                  [&partialCombination](const std::vector<size_t> &group1,
                                        const std::vector<size_t> &group2) {
                    return compareGroupsBelongToSameArray(
                      group1, group2, partialCombination, true);
                  },
                  [&partialCombination](const std::vector<size_t> &group1,
                                        const std::vector<size_t> &group2) {
                    return compareGroupsBelongToSameArray(
                      group1, group2, partialCombination, false);
                  });
  return isValid;
}
 
Match::Match(const PlaceholderSet &ps, const std::vector<CandidateDim> &candidates) {
  size_t size = std::distance(ps.begin(), ps.end());
  if (size != candidates.size())
    llvm_unreachable("expect same number for placeholder and candidates");
  size_t index = 0;
  for (const auto &candidate : candidates) {
    placeholderValues_.emplace_back(ps.placeholders_[index++].id_, candidate);
  }
}

MatchCandidate Match::operator[](const Placeholder<FixedOutDimPattern> &placeholder) const {
  MatchCandidate res;
  for (const auto &key : placeholderValues_) {
    if (key.first == placeholder.id_) {
      if (!res.assigned_) {
        res.assigned_ = true;
        res.candidateDimension_ = key.second;
      } else if (res.candidateDimension_ == key.second) {
        llvm_unreachable("different candidate for the same placeholder");
      }
    }
  }
  if (!res.assigned_) {
    llvm_unreachable("not match for the placeholder");
  }
  return res;
}

bool CandidateDim::operator==(const CandidateDim &candidate) const {
  if (this->inputDimPos_ != candidate.inputDimPos_) {
    return false;
  }
  if (this->operation_ != candidate.operation_) {
    return false;
  }
  return true;
}

bool CandidateDim::operator!=(const CandidateDim &candidate) const {
  if (this->inputDimPos_ == candidate.inputDimPos_) {
    return false;
  }
  if (this->operation_ == candidate.operation_) {
    return false;
  }
  return true;
}

template <>
void Placeholder<FixedOutDimPattern>::dump() {
  outs() << "---> Placeholder<FixedOutDimPattern>\n";
  outs() << "constant: " << pattern_.affine_.constant_ << "\n"; 
  outs() << "coefficient: " << pattern_.affine_.coefficient_ << "\n";
  outs() << "id: " << id_ << "\n";
  outs() << "#Candidates: " << candidates_.size() << "\n";
  if (candidates_.size()) {
    for (size_t i = 0; i < candidates_.size() - 1; i++) {
      outs() << candidates_[i].inputDimPos_;
      outs() << " - ";
    }
    outs() << candidates_[candidates_.size() - 1].inputDimPos_;
    outs() << "\n";
    outs() << "operations" << "\n";
    for (size_t i = 0; i < candidates_.size() - 1; i++) {
      outs() << candidates_[i].operation_;
      outs() << " - ";
    }
    outs() << candidates_[candidates_.size() - 1].operation_;
    outs() << "\n";
  }
}

template <>
void Placeholder<char>::dump() {
  outs() << "---> Placeholder<char>\n";
}

// dump placeholderSet.
void PlaceholderSet::dump() {
  outs() << "#Placeholder: " << placeholders_.size() << "\n";
  outs() << "Placeholder breakdown: " << "\n";
  if (!placeholders_.size())
    outs() << "empty\n";
  else {
    for (size_t i = 0; i < placeholders_.size(); i++)
      placeholders_[i].dump();
  }
  outs() << "#Groups: " << placeholderGroups_.size() << "\n";
  outs() << "Groups breakdown: " << "\n";
  if (!placeholderGroups_.size())
    outs() << "empty\n";
  else {
    for (const auto &group : placeholderGroups_) {
      for (size_t i = 0; i < group.size() - 1; i++) {
        outs() << group[i] << "-";
      }
      outs() << group[group.size() - 1] << "\n";
    }
  }
  outs() << "#Folds: " << placeholderFolds_.size() << "\n";
  outs() << "Folds breakdown: " << "\n";
  if (!placeholderFolds_.size()) {
    outs() << "empty\n";
  }
  else {
    for (size_t i = 0; i < placeholderFolds_.size() - 1; i++) {
      outs() << placeholderFolds_[i] << "-";
    }
    outs() << placeholderFolds_[placeholderFolds_.size() - 1] << "\n";
  }
  outs() << "\n\n";
}

void PlaceholderGroupedSet::dump() {
  PlaceholderSet::dump();
  outs() << "Group Folds: " << "\n";
  if (placeholderGroupFolds_.size()) {
    for (size_t i = 0; i < placeholderGroupFolds_.size() - 1; i++) {
      llvm::outs() << placeholderGroupFolds_[i] << " - ";
    }
    outs() << placeholderGroupFolds_[placeholderGroupFolds_.size() - 1] << "\n";
  }
}

} // end namespace looptactics.
