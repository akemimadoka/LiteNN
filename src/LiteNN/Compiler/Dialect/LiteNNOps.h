#ifndef LITENN_COMPILER_DIALECT_LITENNOPS_H
#define LITENN_COMPILER_DIALECT_LITENNOPS_H

#include "Dialect/LiteNNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/LiteNNEnums.h.inc"

#define GET_OP_CLASSES
#include "Dialect/LiteNNOps.h.inc"

#endif
