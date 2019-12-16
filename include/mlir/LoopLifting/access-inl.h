#include "llvm/Support/raw_ostream.h"

namespace looptactics {

template <typename PatternPayload>
thread_local size_t Placeholder<PatternPayload>::nextId_ = 0;

template<class T, class...>
struct are_same : std::true_type
{};

template<class T, class U, class... TT>
struct are_same<T, U, TT...>
    : std::integral_constant<bool, std::is_same<T,U>{} && are_same<T, TT...>{}>
{};

// Set up the folder list based on the placeholder position.
// A placeholder fold is an identifier of a set of placeholders
// that must get assigned the same candidate.
// For example, given access(_i, _j) and access(_j, _i), the 
// folder list is: 0-1-1-0. 
template <typename Iterator>
static void setUpFolds(const Iterator &iterable, std::vector<size_t> &folds) {
  folds.clear();
  size_t size = std::distance(iterable.begin(), iterable.end());
  folds.reserve(size);
  std::vector<std::pair<size_t, size_t>> knownIds;  
  size_t index = 0;
  for (const auto &placeholder : iterable) {
    auto namePos = std::find_if(knownIds.begin(), knownIds.end(),
      [&placeholder] (const std::pair<size_t, size_t> &pair) {
        return pair.first == placeholder.id_;
      });
    if (namePos == knownIds.end()) {
      knownIds.emplace_back(placeholder.id_, index);
      folds.emplace_back(index);
    }
    else {
      folds.emplace_back(namePos->second);
    }
    ++index;  
  }
}

// Create a single access for fixedOutDimPatterns.
template <typename... Args>
std::vector<Placeholder<FixedOutDimPattern>> access(Args... args) {
  static_assert(are_same<Placeholder<FixedOutDimPattern>, Args...>{},
    "all args must be of type Placeholder<FixedOutDimPattern>");
  int pos = 0;
  std::vector<Placeholder<FixedOutDimPattern>> results{};
  for (auto placeholder : {args...}) {
    placeholder.pattern_.outDimPos_ = pos++;
    results.emplace_back(placeholder);
  }
  return results;
}

// Create a single access for an array access and
// fixedOutDimPatterns. 
template <typename... Args>
ArrayPlaceholderList access(ArrayPlaceholder array, Args... args) {
  return {array, {access(args...)}};
}

// Create a placeholderSet object.
template <typename... Args>
PlaceholderSet allOf(Args... args) {
  static_assert(are_same<std::vector<Placeholder<FixedOutDimPattern>>, Args...>{},
    "all args must be of type std::vector<Placeholder<FixedOutDimPattern>>");
  PlaceholderSet ps;
  std::vector<std::vector<Placeholder<FixedOutDimPattern>>> placeholderLists = {args...};
  for (const auto &placeholderList : placeholderLists) {
    if (placeholderList.empty()) {
      continue;
    }
    size_t index = std::distance(ps.begin(), ps.end());
    ps.placeholderGroups_.emplace_back();
    for (const auto &placeholder : placeholderList) {
      ps.placeholders_.push_back(placeholder);
      ps.placeholderGroups_.back().push_back(index++);
    }
  }
  setUpFolds(ps, ps.placeholderFolds_);
  return ps;
}

// Create a placeholderGroupSet object.
template <typename... Args>
PlaceholderGroupedSet allOf(ArrayPlaceholderList arg, Args... args) {
  static_assert(are_same<ArrayPlaceholderList, Args...>{},
    "all args must be of type ArrayPlaceholderList");
  auto pgs = PlaceholderGroupedSet(allOf(arg.list, args.list...));
  setUpFolds(std::initializer_list<ArrayPlaceholder>{arg.array, args.array...},
             pgs.placeholderGroupFolds_);
  return pgs;
}

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

static inline Value* getMemRef(Operation *op) {
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

template <typename PlaceholderCollection>
static void recursivelyCheckCombinations(
    PlaceholderCollection &ps, std::vector<CandidateDim> partialCombination,
    Matches &suitableCandidates) {
 // Is the current combination a valid one?
  if (!ps.isSuitableCombination(partialCombination))
    return;
 
  // The combination is known to be valid. We are good
  // to go.
  size_t size = std::distance(ps.begin(), ps.end());
  if (partialCombination.size() == size) {
    suitableCandidates.emplace_back(ps, partialCombination);
    return;
  }

  // If the combination is not yet valid add one new
  // element and recurse.
  auto pos = partialCombination.size();
  for (const auto &candidate : ps.placeholders_[pos].candidates_) {
    partialCombination.push_back(candidate);
    recursivelyCheckCombinations(ps, partialCombination, suitableCandidates);
    partialCombination.pop_back();
  }
}

template <typename PlaceholderCollection>
static Matches suitableCandidates(PlaceholderCollection &ps) {
  Matches result;
  recursivelyCheckCombinations(ps, {}, result);
  return result;
}

template <typename PlaceholderCollection>
static Matches matchImpl(const SmallVector<Operation *, 8> &ops, PlaceholderCollection &ps) {
  if (!ops.size()) {
    return {};
  }
  for (auto &placeholder : ps) {
    for (auto &op : ops) {
      for (auto &&c : FixedOutDimPattern::candidates(op, placeholder.pattern_)) {
        placeholder.candidates_.push_back(c);
      }
    }
    if (placeholder.candidates_.empty()) {
      return {};
    }
  }
  return suitableCandidates(ps);
}

template <typename PlaceholderCollection>
Matches match(const SmallVector<Operation *, 8> &firstOpsSet, PlaceholderCollection psFirst,
    const SmallVector<Operation *, 8> &secondOpsSet, PlaceholderCollection psSecond) {
  if ((!psSecond.placeholders_.size()) && (!secondOpsSet.size())) {
    return matchImpl(firstOpsSet, psFirst);
  }
  else {
    llvm_unreachable("not implemented!");
  }
  return {};
}

} // end namespace looptactics.
