#include <memory>

#include <mlir/Pass/Pass.h>

#ifndef LITENN_COMPILER_PASS_LOWERLITENNPASS_H
#define LITENN_COMPILER_PASS_LOWERLITENNPASS_H

namespace litenn
{

std::unique_ptr<mlir::Pass> createLowerLiteNNPass();

} // namespace litenn

#endif
