namespace matchers {

template<class T, class...>
struct are_same : std::true_type
{};

template<class T, class U, class... TT>
struct are_same<T, U, TT...>
    : std::integral_constant<bool, std::is_same<T,U>{} && are_same<T, TT...>{}>
{};

// Create a single access.
template <typename... Args>
std::vector<Placeholder> access(Args... args) {
  static_assert(are_same<Placeholder, Args...>{},
    "all args must be of type Placeholder");
  int pos = 0;
  std::vector<Placeholder> results{};
  for (auto placeholder : {args...}) {
    placeholder.pattern_.outDimPos_ = pos++;
    results.emplace_back(placeholder);
  }
  return results;
}

// Create a placeholderSet object.
template <typename... Args>
PlaceholderSet allOf(Args... args) {
  static_assert(are_same<std::vector<Placeholder>, Args...>{},
    "all args must be of type std::vector<Placeholder<FixedOutDimPattern>>");
  PlaceholderSet ps;
  std::vector<std::vector<Placeholder>> placeholderLists = {args...};
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

} // end namespace matchers.
