#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace litenn
{

std::unique_ptr<mlir::Pass> createLowerLiteNNPass();

} // namespace litenn
