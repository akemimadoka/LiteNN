#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
	std::string ReadSourceFile(std::string_view relativePath)
	{
		const auto path = std::filesystem::path(LITENN_SOURCE_DIR) / std::filesystem::path(relativePath);
		std::ifstream input(path, std::ios::binary);
		if (!input)
		{
			throw std::runtime_error("Failed to open source file for public API guard: " + path.string());
		}
		std::ostringstream output;
		output << input.rdbuf();
		return output.str();
	}

	struct ForbiddenPattern
	{
		std::string_view file;
		std::string_view pattern;
		std::string_view replacement;
	};
} // namespace

TEST(G14PublicApiGuard, PlanNativeRuntimeEntrypointsDoNotReintroduceGraphOverloads)
{
	const std::vector<ForbiddenPattern> forbidden{
		{ "src/LiteNN/Runtime/Scheduler.h", "BuildRuntimeSchedule(const Graph&", "BuildExecutableModule(graph)" },
		{ "src/LiteNN/Runtime/Placement.h", "BuildPlacementPlan(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Training/TrainStepPlan.h", "BuildTrainStepPlan(const Graph&", "BuildExecutableModule(graph)" },
		{ "src/LiteNN/Compiler/CompiledModule.h", "CompileArtifact(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Compiler/CompiledModule.h", "Compile(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Compiler/Dump.h", "DumpMLIR(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Compiler/Translation/GraphToMLIR.h", "translateGraphToMLIR", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunSubgraph(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunForward(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunForwardWithTrace(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunBackward(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Layer/Linear.h", "CreateLinear(Graph&", "CreateLinear(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Linear.h", "BuildLinear(Graph&", "BuildLinear(ModelBuilder&)" },
		{ "src/LiteNN/Layer/LayerNorm.h", "CreateLayerNorm(Graph&", "CreateLayerNorm(ModelBuilder&)" },
		{ "src/LiteNN/Layer/RMSNorm.h", "CreateRMSNorm(Graph&", "CreateRMSNorm(ModelBuilder&)" },
		{ "src/LiteNN/Layer/RMSNorm.h", "BuildRMSNorm(Graph&", "BuildRMSNorm(ModelBuilder&)" },
		{ "src/LiteNN/Layer/SwiGLU.h", "CreateSwiGLUMLP(Graph&", "CreateSwiGLUMLP(ModelBuilder&)" },
		{ "src/LiteNN/Layer/SwiGLU.h", "BuildSwiGLUMLP(Graph&", "BuildSwiGLUMLP(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildReLU(Graph&", "BuildReLU(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Arange.h", "BuildArange(Graph&", "BuildArange(ModelBuilder&)" },
		{ "src/LiteNN/Layer/AddId.h", "BuildAddId(Graph&", "BuildAddId(ModelBuilder&)" },
		{ "src/LiteNN/Layer/MulMatId.h", "BuildMulMatId(Graph&", "BuildMulMatId(ModelBuilder&)" },
	};

	for (const auto& entry : forbidden)
	{
		const auto text = ReadSourceFile(entry.file);
		EXPECT_EQ(text.find(entry.pattern), std::string::npos)
		    << entry.file << " must stay plan/module-native; callers should pass " << entry.replacement
		    << " explicitly at the migration boundary";
	}
}
