#include <type_traits>
#include "access.h"

namespace looptactics {

template <typename OpType>
inline Matcher<OpType> m_Loop(std::function<bool(OpType op)> callback) {
  if ((!std::is_same<OpType, loop::ForOp>::value) &&
      (!std::is_same<OpType, AffineForOp>::value))
    llvm_unreachable("cannot create a non type loop for a loop matcher");
  Matcher<OpType> m;
  m.callback_ = callback;
  return m;
}

template <typename OpType>
inline Matcher<OpType> m_Loop(Matcher<OpType> &&child) {
  if ((!std::is_same<OpType, loop::ForOp>::value) &&
      (!std::is_same<OpType, AffineForOp>::value))
    llvm_unreachable("cannot create a non type loop for a loop matcher");
  Matcher<OpType> m;
  m.children_.emplace_back(child);
  return m;
}

template <typename OpType>
bool isMatching(const Matcher<OpType> &matcher, Operation *op) {
  if (!op)
    return false; 
  if (!matcher.hasSameType(op))
    return false;
  if (matcher.callback_ && !matcher.callback_(dyn_cast<OpType>(op)))
    return false;
  size_t nChildren = matcher.children_.size();
  for (size_t i = 0; i < nChildren; i++) 
    if (!isMatching(matcher.children_.at(i), &op->getRegion(0).front().front())) 
      return false; 

  return true;
}

static inline bool isStore(Operation *op) {
  if ((isa<AffineStoreOp>(op)) /*|| (isa<LLVM::StoreOp>(op))*/)
    return true;
  return false;
}

static inline bool isLoad(Operation *op) {
  if ((isa<AffineLoadOp>(op)) /*|| (isa<LLVM::LoadOp>(op))*/)
    return true;
  return false;
}

template <typename OpType>
static void collectAccesses(OpType &forOp, SmallVector<Operation *, 8> &loadOps,
    SmallVector<Operation *, 8> &storeOps) {
  forOp.getOperation()->walk([&] (Operation *op) {
    if (isStore(op))
      storeOps.push_back(op);
    if (isLoad(op))
      loadOps.push_back(op);
  });
}

template <typename OpType>
static bool hasGemmOperation(OpType op) {

  return true;
}

static inline bool hasGemmAccessPattern(const SmallVector<Operation *, 8> &loads,
    const SmallVector<Operation *, 8> &stores) {
  auto _i = placeholder();
  auto _j = placeholder();
  auto _k = placeholder();
  auto psRead = allOf(access(_i, _j), access(_i, _k), access(_k, _j));
  auto psWrite = allOf(access(_i, _j));
  auto matches = match(stores, psWrite, loads, psRead);
  
  return (matches.size() == 1);
}  

template <typename OpType>
static bool hasGemmPatternImpl(OpType op) {
  SmallVector<Operation *, 8> loadOps;
  SmallVector<Operation *, 8> storeOps;
  collectAccesses(op, loadOps, storeOps);
  if (loadOps.size() != 3)
    return false;
  if (storeOps.size() != 1)
    return false;
  return hasGemmAccessPattern(loadOps, storeOps) &&
    hasGemmOperation(op);
}

template <typename OpType>
std::function<bool(OpType)> hasGemmPattern = [](OpType op) {
  return hasGemmPatternImpl(op);
};

} // end namespace looptactics.
