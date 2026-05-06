#include "Dialect/LiteNNOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace litenn;

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection& symbolTable)
{
	auto fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, getCalleeAttr());
	if (!fn)
		return emitOpError("references unknown function '") << getCallee() << "'";
	return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder& builder, OperationState& state, llvm::StringRef name, FunctionType type)
{
	state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
	state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
	state.addRegion();
}

ParseResult FuncOp::parse(OpAsmParser& parser, OperationState& result)
{
	auto buildFuncType = [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
	                        function_interface_impl::VariadicFlag, std::string&) {
		return builder.getFunctionType(argTypes, results);
	};
	return function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false,
	                                                getFunctionTypeAttrName(result.name), buildFuncType,
	                                                getArgAttrsAttrName(result.name),
	                                                getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter& p)
{
	function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
	                                         getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor)
{
	return getValueAttr();
}

//===----------------------------------------------------------------------===//
// CondOp
//===----------------------------------------------------------------------===//

ParseResult CondOp::parse(OpAsmParser& parser, OperationState& result)
{
	// litenn.cond %cond, %args... : type_cond, (types_args...) -> (types_results...)
	//   then { ... }
	//   else { ... }
	OpAsmParser::UnresolvedOperand condOperand;
	SmallVector<OpAsmParser::UnresolvedOperand> argOperands;
	Type condType;
	SmallVector<Type> argTypes;
	SmallVector<Type> resultTypes;

	if (parser.parseOperand(condOperand))
		return failure();

	if (succeeded(parser.parseOptionalComma()))
	{
		if (parser.parseOperandList(argOperands))
			return failure();
	}

	if (parser.parseColon() || parser.parseType(condType))
		return failure();

	if (succeeded(parser.parseOptionalComma()))
	{
		if (parser.parseLParen() || parser.parseTypeList(argTypes) || parser.parseRParen())
			return failure();
	}

	if (parser.parseArrow())
		return failure();

	if (parser.parseLParen() || parser.parseTypeList(resultTypes) || parser.parseRParen())
		return failure();

	if (parser.resolveOperand(condOperand, condType, result.operands))
		return failure();

	if (parser.resolveOperands(argOperands, argTypes, parser.getCurrentLocation(), result.operands))
		return failure();

	result.addTypes(resultTypes);

	auto* thenRegion = result.addRegion();
	auto* elseRegion = result.addRegion();

	if (parser.parseRegion(*thenRegion) || parser.parseKeyword("else") || parser.parseRegion(*elseRegion))
		return failure();

	if (parser.parseOptionalAttrDict(result.attributes))
		return failure();

	return success();
}

void CondOp::print(OpAsmPrinter& p)
{
	p << ' ' << getCondition();
	if (!getArgs().empty())
	{
		p << ", ";
		p.printOperands(getArgs());
	}
	p << " : " << getCondition().getType();
	if (!getArgs().empty())
	{
		p << ", (";
		llvm::interleaveComma(getArgs().getTypes(), p);
		p << ')';
	}
	p << " -> (";
	llvm::interleaveComma(getResultTypes(), p);
	p << ") ";
	p.printRegion(getThenRegion());
	p << " else ";
	p.printRegion(getElseRegion());
	p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ParseResult WhileOp::parse(OpAsmParser& parser, OperationState& result)
{
	// litenn.while %init_args... : (types...) -> (types...)
	//   cond { ... }
	//   body { ... }
	SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
	SmallVector<Type> initTypes;
	SmallVector<Type> resultTypes;

	if (parser.parseOperandList(initOperands) || parser.parseColon() || parser.parseLParen() ||
	    parser.parseTypeList(initTypes) || parser.parseRParen() || parser.parseArrow() || parser.parseLParen() ||
	    parser.parseTypeList(resultTypes) || parser.parseRParen())
		return failure();

	if (parser.resolveOperands(initOperands, initTypes, parser.getCurrentLocation(), result.operands))
		return failure();

	result.addTypes(resultTypes);

	auto* condRegion = result.addRegion();
	auto* bodyRegion = result.addRegion();

	if (parser.parseKeyword("cond") || parser.parseRegion(*condRegion) || parser.parseKeyword("body") ||
	    parser.parseRegion(*bodyRegion))
		return failure();

	if (parser.parseOptionalAttrDict(result.attributes))
		return failure();

	return success();
}

void WhileOp::print(OpAsmPrinter& p)
{
	p << ' ';
	p.printOperands(getInitArgs());
	p << " : (";
	llvm::interleaveComma(getInitArgs().getTypes(), p);
	p << ") -> (";
	llvm::interleaveComma(getResultTypes(), p);
	p << ") cond ";
	p.printRegion(getCondRegion());
	p << " body ";
	p.printRegion(getBodyRegion());
	p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// FusedOp
//===----------------------------------------------------------------------===//

ParseResult FusedOp::parse(OpAsmParser& parser, OperationState& result)
{
	// litenn.fused pattern(%args...) : (types...) -> (types...) { body }
	StringRef patternStr;
	SmallVector<OpAsmParser::UnresolvedOperand> argOperands;
	SmallVector<Type> argTypes;
	SmallVector<Type> resultTypes;

	// Parse pattern keyword as enum value
	if (parser.parseKeyword(&patternStr))
		return failure();

	auto patternKind = symbolizeFusionPatternKind(patternStr);
	if (!patternKind)
		return failure();
	result.addAttribute("pattern",
		FusionPatternKindAttr::get(parser.getContext(), *patternKind));

	if (parser.parseLParen() || parser.parseOperandList(argOperands) || parser.parseRParen() || parser.parseColon() ||
	    parser.parseLParen() || parser.parseTypeList(argTypes) || parser.parseRParen() || parser.parseArrow() ||
	    parser.parseLParen() || parser.parseTypeList(resultTypes) || parser.parseRParen())
		return failure();

	if (parser.resolveOperands(argOperands, argTypes, parser.getCurrentLocation(), result.operands))
		return failure();

	result.addTypes(resultTypes);

	auto* bodyRegion = result.addRegion();
	if (parser.parseRegion(*bodyRegion))
		return failure();

	if (parser.parseOptionalAttrDict(result.attributes))
		return failure();

	return success();
}

void FusedOp::print(OpAsmPrinter& p)
{
	p << ' ' << getPattern() << '(';
	p.printOperands(getArgs());
	p << ") : (";
	llvm::interleaveComma(getArgs().getTypes(), p);
	p << ") -> (";
	llvm::interleaveComma(getResultTypes(), p);
	p << ") ";
	p.printRegion(getBody());
	p.printOptionalAttrDict((*this)->getAttrs(), {"pattern"});
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/LiteNNOps.cpp.inc"
