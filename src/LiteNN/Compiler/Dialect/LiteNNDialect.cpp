#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"

#include "Dialect/LiteNNDialect.cpp.inc"
#include "Dialect/LiteNNEnums.cpp.inc"

namespace litenn
{

	void LiteNNDialect::initialize()
	{
		addOperations<
#define GET_OP_LIST
#include "Dialect/LiteNNOps.cpp.inc"
		    >();
	}

} // namespace litenn
