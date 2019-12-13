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

// Create a single access.
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
  ps.setUpFolds();
  return ps;
}

template <typename PatternPayload>
void Placeholder<PatternPayload>::dump() {
  if (!std::is_same<PatternPayload, Placeholder<FixedOutDimPattern>>::value)
    return;
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


} // end namespace looptactics.
