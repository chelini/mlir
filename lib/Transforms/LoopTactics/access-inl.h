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
