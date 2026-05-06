#ifndef LITENN_OPS_TD
#define LITENN_OPS_TD

include "Dialect/LiteNNDialect.td"

//===----------------------------------------------------------------------===//
// FuncOp — represents a Subgraph
//===----------------------------------------------------------------------===//

def LiteNN_FuncOp : LiteNN_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove, Symbol
]> {
  let summary = "LiteNN subgraph function";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::FunctionType":$type)>
  ];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//
    mlir::Region *getCallableRegion() { return &getBody(); }
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
  }];
}

def LiteNN_ReturnOp : LiteNN_Op<"return", [
    Pure, HasParent<"FuncOp">, Terminator, ReturnLike
]> {
  let summary = "return values from a litenn.func";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Variable (global tensor) operations
//===----------------------------------------------------------------------===//

def LiteNN_VariableOp : LiteNN_Op<"variable", [Symbol]> {
  let summary = "global trainable variable declaration";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    OptionalAttr<ElementsAttr>:$initial_value
  );
  let assemblyFormat = "$sym_name `:` $type (`=` $initial_value^)? attr-dict";
}

def LiteNN_GetVariableOp : LiteNN_Op<"get_variable", [Pure]> {
  let summary = "load a trainable variable by symbol reference";
  let arguments = (ins FlatSymbolRefAttr:$variable);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$variable attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

def LiteNN_ConstantOp : LiteNN_Op<"constant", [Pure, ConstantLike]> {
  let summary = "tensor constant";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "attr-dict $value `:` type($result)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Unary operation
//===----------------------------------------------------------------------===//

def LiteNN_UnaryOp : LiteNN_Op<"unary", [Pure]> {
  let summary = "elementwise unary operation";
  let arguments = (ins
    LiteNN_UnaryOpKind:$op,
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$op `(` $input `)` attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Binary operation
//===----------------------------------------------------------------------===//

def LiteNN_BinaryOp : LiteNN_Op<"binary", [Pure]> {
  let summary = "elementwise binary operation (or matmul)";
  let arguments = (ins
    LiteNN_BinaryOpKind:$op,
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$op `(` $lhs `,` $rhs `)` attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Cast
//===----------------------------------------------------------------------===//

def LiteNN_CastOp : LiteNN_Op<"cast", [Pure]> {
  let summary = "type cast";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Call
//===----------------------------------------------------------------------===//

def LiteNN_CallOp : LiteNN_Op<"call", [
    CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "call another litenn.func subgraph";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)";
  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", mlir::cast<mlir::SymbolRefAttr>(callee.get<mlir::SymbolRefAttr>()));
    }
    mlir::Operation::operand_range getArgOperands() {
      return getOperands();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Cond (conditional branch with regions)
//===----------------------------------------------------------------------===//

def LiteNN_CondOp : LiteNN_Op<"cond", [RecursiveMemoryEffects]> {
  let summary = "conditional execution";
  let arguments = (ins
    AnyTensor:$condition,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>);
  let regions = (region
    SizedRegion<1>:$then_region,
    SizedRegion<1>:$else_region
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// While (functional loop with regions)
//===----------------------------------------------------------------------===//

def LiteNN_WhileOp : LiteNN_Op<"while", [RecursiveMemoryEffects]> {
  let summary = "functional while loop";
  let arguments = (ins Variadic<AnyType>:$init_args);
  let results = (outs Variadic<AnyType>);
  let regions = (region
    SizedRegion<1>:$cond_region,
    SizedRegion<1>:$body_region
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Yield (region terminator for cond/while/fused)
//===----------------------------------------------------------------------===//

def LiteNN_YieldOp : LiteNN_Op<"yield", [Pure, Terminator, ReturnLike]> {
  let summary = "yield values from a region";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Reduce
//===----------------------------------------------------------------------===//

def LiteNN_ReduceOp : LiteNN_Op<"reduce", [Pure]> {
  let summary = "reduction along an axis";
  let arguments = (ins
    LiteNN_ReduceOpKind:$op,
    AnyTensor:$input,
    I64Attr:$axis
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$op `(` $input `,` `axis` `=` $axis `)` attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Reshape
//===----------------------------------------------------------------------===//

def LiteNN_ReshapeOp : LiteNN_Op<"reshape", [Pure]> {
  let summary = "reshape tensor (target shape in result type)";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Concat
//===----------------------------------------------------------------------===//

def LiteNN_ConcatOp : LiteNN_Op<"concat", [Pure]> {
  let summary = "concatenate tensors along axis";
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    I64Attr:$axis
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$inputs `,` `axis` `=` $axis attr-dict `:` `(` type($inputs) `)` `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Slice
//===----------------------------------------------------------------------===//

def LiteNN_SliceOp : LiteNN_Op<"slice", [Pure]> {
  let summary = "slice along axis";
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis,
    I64Attr:$start,
    I64Attr:$length
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input `[` `axis` `=` $axis `,` $start `,` $length `]` attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Fused operation
//===----------------------------------------------------------------------===//

def LiteNN_FusedOp : LiteNN_Op<"fused", [RecursiveMemoryEffects]> {
  let summary = "fused operation with body region";
  let arguments = (ins
    LiteNN_FusionPatternKind:$pattern,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

#endif // LITENN_OPS_TD
