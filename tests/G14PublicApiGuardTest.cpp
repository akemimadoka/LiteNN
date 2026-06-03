#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
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

	bool EndsWith(std::string_view value, std::string_view suffix)
	{
		return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
	}
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
		{ "src/LiteNN/Compiler/Translation/GraphToMLIR.cpp", "BuildMLIRGraphFromPlan",
		  "direct ExecutablePlan lowering" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunSubgraph(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunForward(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunForwardWithTrace(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunBackward(const Graph&", "BuildExecutablePlan(graph)" },
		{ "src/LiteNN/Serialization/ModelIO.h", "SaveModel(const Graph&", "SaveGraphArchive(graph)" },
		{ "src/LiteNN/Serialization/ModelIO.h", "SaveModelExternalWeights(const Graph&",
		  "SaveGraphArchiveExternalWeights(graph)" },
		{ "src/LiteNN/Serialization/ModelIO.h", "inline Graph LoadModel(", "LoadGraphArchive(path)" },
		{ "src/LiteNN/Serialization/ModelIO.h", "kModelMagic", "kGraphArchiveMagic" },
		{ "src/LiteNN/Serialization/ModelIO.h", "kModelVersion", "kGraphArchiveVersion" },
		{ "src/LiteNN/Serialization/ModelIO.h", "enum class NodeKind", "enum class GraphArchiveNodeKind" },
		{ "src/LiteNN/Serialization/ModelPackageIO.h", "#include <simdjson.h>",
		  "private implementation parsing in ModelPackageIO.cpp" },
		{ "src/LiteNN/Training/Trainer.h", "#include <LiteNN/Compiler/CompiledModule.h>",
		  "TrainStepAOTRunner in LiteNNCompiler" },
		{ "src/LiteNN/Training/Trainer.h", "Trainer AOT execution policy is not wired yet",
		  "explicit forward runner plus train-step diagnostic" },
		{ "src/LiteNN/Layer/Linear.h", "CreateLinear(Graph&", "CreateLinear(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Linear.h", "BuildLinear(Graph&", "BuildLinear(ModelBuilder&)" },
		{ "src/LiteNN/Layer/LayerNorm.h", "CreateLayerNorm(Graph&", "CreateLayerNorm(ModelBuilder&)" },
		{ "src/LiteNN/Layer/RMSNorm.h", "CreateRMSNorm(Graph&", "CreateRMSNorm(ModelBuilder&)" },
		{ "src/LiteNN/Layer/RMSNorm.h", "BuildRMSNorm(Graph&", "BuildRMSNorm(ModelBuilder&)" },
		{ "src/LiteNN/Layer/SwiGLU.h", "CreateSwiGLUMLP(Graph&", "CreateSwiGLUMLP(ModelBuilder&)" },
		{ "src/LiteNN/Layer/SwiGLU.h", "BuildSwiGLUMLP(Graph&", "BuildSwiGLUMLP(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildReLU(Graph&", "BuildReLU(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildGELUErf(Graph&", "BuildGELUErf(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildSigmoid(Graph&", "BuildSigmoid(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildTanh(Graph&", "BuildTanh(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildSiLU(Graph&", "BuildSiLU(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildGELU(Graph&", "BuildGELU(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildELU(Graph&", "BuildELU(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildClamp(Graph&", "BuildClamp(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildLeakyReLU(Graph&", "BuildLeakyReLU(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildHardSigmoid(Graph&", "BuildHardSigmoid(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildHardSwish(Graph&", "BuildHardSwish(ModelBuilder&)" },
		{ "src/LiteNN/Layer/Activation.h", "BuildGELUQuick(Graph&", "BuildGELUQuick(ModelBuilder&)" },
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

TEST(G14PublicApiGuard, PublicLayerBuildHelpersDoNotAcceptRawGraph)
{
	const auto layerDir = std::filesystem::path(LITENN_SOURCE_DIR) / "src" / "LiteNN" / "Layer";
	const std::regex publicBuildWithGraph(R"(inline\s+SubgraphId\s+(Build[A-Za-z0-9_]+)\s*\(\s*Graph&)");
	std::vector<std::string> violations;

	for (const auto& entry : std::filesystem::directory_iterator(layerDir))
	{
		if (!entry.is_regular_file() || entry.path().extension() != ".h")
		{
			continue;
		}

		const auto relative = std::filesystem::path("src") / "LiteNN" / "Layer" / entry.path().filename();
		const auto text = ReadSourceFile(relative.generic_string());
		for (std::sregex_iterator match(text.begin(), text.end(), publicBuildWithGraph), end; match != end; ++match)
		{
			const auto helperName = (*match)[1].str();
			if (EndsWith(helperName, "Impl"))
			{
				continue;
			}
			violations.push_back(relative.generic_string() + ": " + helperName + "(Graph&)");
		}
	}

	EXPECT_TRUE(violations.empty()) << "Public layer Build* helpers must accept ModelBuilder& after vNext:\n"
	                               << [&]() {
		                                  std::ostringstream output;
		                                  for (const auto& violation : violations)
		                                  {
			                                  output << violation << '\n';
		                                  }
		                                  return output.str();
	                                  }();
}
