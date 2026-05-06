#ifndef LITENN_DIALECT_TD
#define LITENN_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

def LiteNN_Dialect : Dialect {
  let name = "litenn";
  let cppNamespace = "::litenn";
  let summary = "LiteNN neural network dialect";
}

class LiteNN_Op<string mnemonic, list<Trait> traits = []>
    : Op<LiteNN_Dialect, mnemonic, traits>;

// ---- Enum Attrs ----

def LiteNN_UnaryOpKind : I32EnumAttr<"UnaryOpKind", "litenn unary operation", [
  I32EnumAttrCase<"Negate",           0>,
  I32EnumAttrCase<"Abs",              1>,
  I32EnumAttrCase<"Sqrt",             2>,
  I32EnumAttrCase<"Exp",              3>,
  I32EnumAttrCase<"Log",              4>,
  I32EnumAttrCase<"Sin",              5>,
  I32EnumAttrCase<"Cos",              6>,
  I32EnumAttrCase<"Tan",              7>,
  I32EnumAttrCase<"Arcsin",           8>,
  I32EnumAttrCase<"Arccos",           9>,
  I32EnumAttrCase<"Arctan",          10>,
  I32EnumAttrCase<"Transpose",       11>,
  I32EnumAttrCase<"LogicalNegation", 12>
]> {
  let cppNamespace = "::litenn";
}

def LiteNN_BinaryOpKind : I32EnumAttr<"BinaryOpKind", "litenn binary operation", [
  I32EnumAttrCase<"Add",       0>,
  I32EnumAttrCase<"Subtract",  1>,
  I32EnumAttrCase<"Multiply",  2>,
  I32EnumAttrCase<"Divide",    3>,
  I32EnumAttrCase<"MatMul",    4>,
  I32EnumAttrCase<"Pow",       5>,
  I32EnumAttrCase<"Max",       6>,
  I32EnumAttrCase<"Min",       7>,
  I32EnumAttrCase<"Less",      8>,
  I32EnumAttrCase<"Greater",   9>,
  I32EnumAttrCase<"Equal",    10>
]> {
  let cppNamespace = "::litenn";
}

def LiteNN_ReduceOpKind : I32EnumAttr<"ReduceOpKind", "litenn reduce operation", [
  I32EnumAttrCase<"Sum",  0>,
  I32EnumAttrCase<"Mean", 1>,
  I32EnumAttrCase<"Max",  2>
]> {
  let cppNamespace = "::litenn";
}

def LiteNN_FusionPatternKind : I32EnumAttr<"FusionPatternKind", "litenn fusion pattern", [
  I32EnumAttrCase<"MatMulBiasAdd",    0>,
  I32EnumAttrCase<"ElementWiseChain", 1>
]> {
  let cppNamespace = "::litenn";
}

#endif // LITENN_DIALECT_TD
