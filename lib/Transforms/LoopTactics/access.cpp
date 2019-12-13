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

// Set up the folder list based on the placeholder position.
// A placeholder fold is an identifier of a set of placeholders
// that must get assigned the same candidate.
// For example, given access(_i, _j) and access(_j, _i), the 
// folder list is: 0-1-1-0. 
void PlaceholderSet::setUpFolds() {
  placeholderFolds_.clear();
  placeholderFolds_.reserve(std::distance(begin(), end()));
  std::vector<std::pair<size_t, size_t>> knownIds;
  size_t index = 0;
  for (const auto &placeholder : *this) {
    auto namePos = std::find_if(knownIds.begin(), knownIds.end(),
        [&placeholder] (const std::pair<size_t, size_t> &pair) {
            return pair.first == placeholder.id_;
        });
    if (namePos == knownIds.end()) {
      knownIds.emplace_back(placeholder.id_, index);
      placeholderFolds_.emplace_back(index);
    }
    else {
      placeholderFolds_.emplace_back(namePos->second);
    }
    ++index;
  }
}

// All the placeholders should get different assignments, expect those
// that belong to the same fold.
static bool hasNoDuplicateAssignments(
    PlaceholderSet &ps, std::vector<CandidateDim> partialCombination) {
  auto folds = ps.placeholderFolds_;
  auto size = partialCombination.size();
  if (size > folds.size()) {
    llvm_unreachable("folds are not propery set");
  }

  for (size_t i = 0; i < size; i++) {
    for (size_t j = i + 1; j < size; j++) {
      if (folds[i] == folds[j]) {
        if (partialCombination[i].inputDimPos_ != partialCombination[j].inputDimPos_) {
          return false;
        }
        else {
          continue;
        }
      }
      if (partialCombination[i].inputDimPos_ == partialCombination[j].inputDimPos_) {
        return false;
      }
    }
  }
  return true;
}

// All the placeholders in a group are either not yet matched, or
// matched the same load/store operation. A load/store operation 
// matched in a group is not matched in any previous group.
static bool groupsAreProperlyFormed(
    PlaceholderSet &ps, std::vector<CandidateDim> partialCombination) {
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

static bool isSuitableCombination(
    PlaceholderSet &ps, std::vector<CandidateDim> partialCombination) {
 return hasNoDuplicateAssignments(ps, partialCombination) &&
        groupsAreProperlyFormed(ps, partialCombination);  
}

// Check if the current combination is a good one.
static void recursivelyCheckCombinations(
    PlaceholderSet &ps, std::vector<CandidateDim> partialCombination,
    Matches &suitableCandidates) {

  // Is the current combination a valid one?
  if (!isSuitableCombination(ps, partialCombination))
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

static Matches suitableCandidates(PlaceholderSet &ps) {
  Matches result;
  recursivelyCheckCombinations(ps, {}, result);
  return result;
}

// Check if the placeholder combination satisfies the load/store
// operations provided.
Matches matchImpl(const SmallVector<Operation *, 8> &ops, PlaceholderSet ps) {
  if (!ops.size()) {
    return {};
  }
  for (auto &placeholder : ps) {
    for (auto &op : ops) {
      for (auto && c : FixedOutDimPattern::candidates(op, placeholder.pattern_)) {
        placeholder.candidates_.push_back(c);
      }
    }
    if (placeholder.candidates_.empty()) {
      return {};
    }
  }
  return suitableCandidates(ps);
}

// Placeholder are not reused across different calls of allOf.
// This means that if we write:
//    auto matchesRead = match(reads, allOf(access(_i, _j)));
//    auto matchesWrite = match(writes, allOf(access(_i, _j));
// the first _i or _j may match different candidate with respect
// to the second _i and _j. This force the user to check the matched
// candidates. This function does this for you. 
Matches match (const SmallVector<Operation *, 8> &opsFirst, PlaceholderSet psFirst,
    const SmallVector<Operation *, 8> &opsSecond, PlaceholderSet psSecond) {
  if ((!psSecond.placeholders_.size()) && (!opsSecond.size())) {
    return matchImpl(opsFirst, psFirst);
  }
  if ((psSecond.placeholders_.size()) && (!opsSecond.size())) {
    return {};  
  }
  if ((!psSecond.placeholders_.size()) && (opsSecond.size())) {
    return {};
  }
  auto matchesFirst = matchImpl(opsFirst, psFirst);
  auto matchesSecond = matchImpl(opsSecond, psSecond);
  if (matchesFirst.size() != matchesSecond.size())
    return {};

  for (const auto& matchFirst : matchesFirst) {
    for (const auto& matchSecond : matchesSecond) {
      auto psValuesFirst = matchFirst.placeholderValues_;
      auto psValuesSecond = matchSecond.placeholderValues_;
      for (const auto& psValueFirst : psValuesFirst) {
        for (const auto& psValueSecond : psValuesSecond) {
          if (psValueFirst.first == psValueSecond.first) {
            if (psValueFirst.second != psValueSecond.second) {
              return {};
            }
          }
        }
      }
    }
  }

  for (size_t i = 0; i < matchesFirst.size(); i++) {
    matchesFirst[i].placeholderValues_.insert(
      matchesFirst[i].placeholderValues_.end(), 
      matchesSecond[i].placeholderValues_.begin(), matchesSecond[i].placeholderValues_.end());
  }
  return matchesFirst;
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

} // end namespace looptactics.
