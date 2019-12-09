namespace matchers {

inline OperationMatcher affineFor(std::function<bool(Operation *op)> callback,
    OperationMatcher &&child) {
  OperationMatcher m;
  m.current_ = OperationType::AffineFor;
  m.callback_ = callback;
  m.children_.emplace_back(child);
  return m;
}

// this is a bit odd. 
inline OperationMatcher affineFor(std::function<bool(Operation *op)> callback) {
  OperationMatcher m;
  m.current_ = OperationType::AffineFor;
  m.callback_ = callback;
  return m;
}

inline OperationMatcher affineFor(OperationMatcher &&child) {
  OperationMatcher m;
  m.current_ = OperationType::AffineFor;
  m.children_.emplace_back(child);
  return m;
}



} // end namespace matchers
