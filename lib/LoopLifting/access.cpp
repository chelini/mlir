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
