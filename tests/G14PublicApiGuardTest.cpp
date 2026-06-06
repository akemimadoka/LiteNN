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
		{ "src/LiteNN/ExecutablePlan.h", "BuildExecutablePlan(const Graph&",
		  "Detail::BuildExecutablePlanFromGraph" },
		{ "src/LiteNN/ExecutablePlan.h", "BuildExecutableModule(const Graph&",
		  "Detail::BuildExecutableModuleFromGraph" },
		{ "src/LiteNN/Runtime/Scheduler.h", "BuildRuntimeSchedule(const Graph&", "Detail::BuildExecutableModuleFromGraph(graph)" },
		{ "src/LiteNN/Runtime/Placement.h", "BuildPlacementPlan(const Graph&", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Training/TrainStepPlan.h", "BuildTrainStepPlan(const Graph&", "Detail::BuildExecutableModuleFromGraph(graph)" },
		{ "src/LiteNN/Compiler/CompiledModule.h", "CompileArtifact(const Graph&", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Compiler/CompiledModule.h", "Compile(const Graph&", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Compiler/Dump.h", "DumpMLIR(const Graph&", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Compiler/Translation/GraphToMLIR.h", "translateGraphToMLIR", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Compiler/Translation/GraphToMLIR.cpp", "BuildMLIRGraphFromPlan",
		  "direct ExecutablePlan lowering" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunSubgraph(const Graph&", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunForward(const Graph&", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunForwardWithTrace(const Graph&", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Runtime/Interpreter.h", "RunBackward(const Graph&", "Detail::BuildExecutablePlanFromGraph(graph)" },
		{ "src/LiteNN/Serialization/ModelIO.h", "SaveModel(const Graph&", "SaveGraphArchive(graph)" },
		{ "src/LiteNN/Serialization/ModelIO.h", "SaveModelExternalWeights(const Graph&",
		  "SaveGraphArchiveExternalWeights(graph)" },
		{ "src/LiteNN/Serialization/ModelIO.h", "inline Graph LoadModel(", "LoadGraphArchive(path)" },
		{ "src/LiteNN/Serialization/ModelIO.h", "kModelMagic", "kGraphArchiveMagic" },
		{ "src/LiteNN/Serialization/ModelIO.h", "kModelVersion", "kGraphArchiveVersion" },
		{ "src/LiteNN/Serialization/ModelIO.h", "enum class NodeKind", "enum class GraphArchiveNodeKind" },
		{ "src/LiteNN/Serialization/ModelPackageIO.h", "#include <simdjson.h>",
		  "private implementation parsing in ModelPackageIO.cpp" },
		{ "src/LiteNN/Serialization/ModelPackageIO.cpp", "\\\"opKind\\\"",
		  "schema/attribute op records in the vNext package plan" },
		{ "src/LiteNN/Serialization/ModelPackageIO.cpp", "entryFunction",
		  "named artifact entries in the vNext package artifact ABI" },
		{ "src/LiteNN/VNextPackage.h", "BuildVNextPackageManifest(\n\t    const Graph&",
		  "Detail::BuildExecutableModuleFromGraph(graph) inside internal construction/test adapters" },
		{ "src/LiteNN/Training/Trainer.h", "#include <LiteNN/Compiler/CompiledModule.h>",
		  "TrainStepAOTRunner in LiteNNTrainingAOT" },
		{ "src/LiteNN/Training/Trainer.h", "Trainer(Graph&",
		  "Trainer(ModelGraph&) with explicit unsafe graph access inside the implementation" },
		{ "src/LiteNN/Training/Trainer.h", "Trainer AOT execution policy is not wired yet",
		  "explicit forward runner plus train-step diagnostic" },
		{ "src/LiteNN/Training/Trainer.h", "Optimizer::ZeroGradients(*graph_",
		  "explicit ParameterSet binding" },
		{ "src/LiteNN/Training/Trainer.h", "Optimizer::StoreVariableGradients(*graph_",
		  "explicit ParameterSet binding" },
		{ "src/LiteNN/Training/Trainer.h", "optimizer_.Step(*graph_",
		  "explicit ParameterSet binding" },
		{ "src/LiteNN/Optimizer/OptimizerUtils.h", "ZeroGradients(Graph&", "ZeroGradients(ParameterSet&)" },
		{ "src/LiteNN/Optimizer/OptimizerUtils.h", "StoreVariableGradients(Graph&",
		  "StoreVariableGradients(ParameterSet&)" },
		{ "src/LiteNN/Optimizer/SGD.h", "Step(Graph&", "Step(ParameterSet&)" },
		{ "src/LiteNN/Optimizer/Adam.h", "Step(Graph&", "Step(ParameterSet&)" },
		{ "src/LiteNN/Optimizer/AdamW.h", "Step(Graph&", "Step(ParameterSet&)" },
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
		{ "src/LiteNN/Layer/AddId.h", "namespace LiteNN::Layer", "LiteNN::Compatibility::GGML" },
		{ "src/LiteNN/Layer/MulMatId.h", "BuildMulMatId(Graph&", "BuildMulMatId(ModelBuilder&)" },
		{ "src/LiteNN/Layer/MulMatId.h", "namespace LiteNN::Layer", "LiteNN::Compatibility::GGML" },
		{ "src/LiteNN/Layer/Window.h", "namespace LiteNN::Layer", "LiteNN::Compatibility::GGML" },
		{ "src/LiteNN/Layer/RelativePosition.h", "namespace LiteNN::Layer", "LiteNN::Compatibility::GGML" },
		{ "src/LiteNN/Layer/Repeat.h", "namespace LiteNN::Layer", "LiteNN::Compatibility::GGML" },
		{ "src/LiteNN/Layer/SSMConv.h", "namespace LiteNN::Layer", "LiteNN::Compatibility::GGML" },
	};

	for (const auto& entry : forbidden)
	{
		const auto text = ReadSourceFile(entry.file);
		EXPECT_EQ(text.find(entry.pattern), std::string::npos)
		    << entry.file << " must stay plan/module-native; callers should pass " << entry.replacement
		    << " explicitly at the internal construction/test boundary";
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

TEST(G14PublicApiGuard, GraphArchiveApisAreRemoved)
{
	const auto text = ReadSourceFile("src/LiteNN/Serialization/ModelIO.h");
	EXPECT_EQ(text.find("namespace Migration"), std::string::npos);
	EXPECT_EQ(text.find("} // namespace Migration"), std::string::npos);
	for (const auto* pattern : {
	         "SaveGraphArchive",
	         "LoadGraphArchive",
	         "kGraphArchiveMagic",
	         "kGraphArchiveVersion",
	         "GraphArchiveNodeKind",
	     })
	{
		EXPECT_EQ(text.find(pattern), std::string::npos) << pattern;
	}
}

TEST(G14PublicApiGuard, ProductionExamplesDoNotUseGraphArchiveDetailApis)
{
	const std::vector<std::string_view> files{
		"example/mnist/mnist_common.h",
		"example/mnist/interpreter.cpp",
		"example/mnist/aot.cpp",
		"example/gguf/conversion_example.cpp",
	};
	for (const auto file : files)
	{
		const auto text = ReadSourceFile(file);
		EXPECT_EQ(text.find("Serialization::Detail::SaveGraphArchive"), std::string::npos) << file;
		EXPECT_EQ(text.find("Serialization::Detail::LoadGraphArchive"), std::string::npos) << file;
	}
}

TEST(G14PublicApiGuard, CompiledTensorSpecUsesTensorTypeContract)
{
	const auto text = ReadSourceFile("src/LiteNN/Compiler/CompiledModule.h");
	const auto specBegin = text.find("struct CompiledTensorSpec");
	const auto imageBegin = text.find("struct CompiledModuleImage");
	ASSERT_NE(specBegin, std::string::npos);
	ASSERT_NE(imageBegin, std::string::npos);
	const auto specBody = text.substr(specBegin, imageBegin - specBegin);

	EXPECT_NE(specBody.find("TensorType type"), std::string::npos);
	EXPECT_EQ(specBody.find("DataType dtype"), std::string::npos);
	EXPECT_EQ(specBody.find("std::vector<std::size_t> shape"), std::string::npos);
	EXPECT_EQ(specBody.find("ShapeView{ shape }"), std::string::npos);

	const auto externalBegin = text.find("struct CompiledModuleExternalTensorInfo");
	const auto optionsBegin = text.find("struct CompilerOptions");
	ASSERT_NE(externalBegin, std::string::npos);
	ASSERT_NE(optionsBegin, std::string::npos);
	const auto externalBody = text.substr(externalBegin, optionsBegin - externalBegin);

	EXPECT_NE(externalBody.find("TensorType type"), std::string::npos);
	EXPECT_EQ(externalBody.find("DataType dtype"), std::string::npos);
	EXPECT_EQ(externalBody.find("std::vector<std::size_t> shape"), std::string::npos);

	const auto safetensorsText = ReadSourceFile("src/LiteNN/Serialization/Safetensors.h");
	const auto tensorInfoBegin = safetensorsText.find("struct SafetensorsTensorInfo");
	const auto importOptionsBegin = safetensorsText.find("struct SafetensorsImportOptions");
	ASSERT_NE(tensorInfoBegin, std::string::npos);
	ASSERT_NE(importOptionsBegin, std::string::npos);
	const auto tensorInfoBody = safetensorsText.substr(tensorInfoBegin, importOptionsBegin - tensorInfoBegin);

	EXPECT_NE(tensorInfoBody.find("TensorType type"), std::string::npos);
	EXPECT_EQ(tensorInfoBody.find("DataType dtype"), std::string::npos);
	EXPECT_EQ(tensorInfoBody.find("std::vector<std::size_t> shape"), std::string::npos);
}

TEST(G14PublicApiGuard, StableBoundaryHeadersKeepTensorTypeAsShapeTypeContract)
{
	const std::vector<std::string_view> headers{
		"src/LiteNN/Compiler/CompiledModule.h",
		"src/LiteNN/Compiler/Dump.h",
		"src/LiteNN/Runtime/Placement.h",
		"src/LiteNN/Runtime/Scheduler.h",
		"src/LiteNN/Serialization/ModelPackageIO.h",
		"src/LiteNN/Serialization/Safetensors.h",
		"src/LiteNN/Serialization/TorchManifest.h",
	};
	const std::vector<std::string_view> forbidden{
		"OutputInfo",
		" TensorSpec",
		"<TensorSpec",
		"std::vector<TensorSpec>",
		"DataType dtype",
		"std::vector<std::size_t> shape",
		"ShapeView shape",
	};

	for (const auto header : headers)
	{
		const auto text = ReadSourceFile(header);
		for (const auto pattern : forbidden)
		{
			EXPECT_EQ(text.find(pattern), std::string::npos)
			    << header << " must expose TensorType as the single shape/type contract, not " << pattern;
		}
	}
}

TEST(G14PublicApiGuard, CoreRuntimeHeadersDoNotPullDeploymentSpecificDependencies)
{
	const std::vector<std::string_view> headers{
		"src/LiteNN/DType.h",
		"src/LiteNN/Device.h",
		"src/LiteNN/ExecutablePlan.h",
		"src/LiteNN/Graph.h",
		"src/LiteNN/MemoryPlan.h",
		"src/LiteNN/ModelBuilder.h",
		"src/LiteNN/OpSchema.h",
		"src/LiteNN/Pass.h",
		"src/LiteNN/Runtime/Placement.h",
		"src/LiteNN/Runtime/Scheduler.h",
		"src/LiteNN/Storage.h",
		"src/LiteNN/Tensor.h",
		"src/LiteNN/TensorType.h",
		"src/LiteNN/Training/StateDict.h",
		"src/LiteNN/Training/TrainStepPlan.h",
		"src/LiteNN/Training/Trainer.h",
	};
	const std::vector<std::string_view> forbiddenIncludes{
		"#include <LiteNN/Compiler/",
		"#include <LiteNN/Device/CUDA.h>",
		"#include <LiteNN/Serialization/Safetensors.h>",
		"#include <LiteNN/Serialization/TorchManifest.h>",
		"#include <LiteNN/Serialization/GGUF",
		"#include <LiteNN/GGUF",
		"#include <simdjson",
		"#include <gguf",
		"#include <cuda",
		"#include <cublas",
		"example/",
	};

	for (const auto header : headers)
	{
		const auto text = ReadSourceFile(header);
		for (const auto include : forbiddenIncludes)
		{
			EXPECT_EQ(text.find(include), std::string::npos)
			    << header << " must stay in the core/runtime distribution boundary and not depend on " << include;
		}
	}
}

TEST(G14PublicApiGuard, CMakeExposesCoreImporterAndFullRuntimeTargets)
{
	const auto text = ReadSourceFile("CMakeLists.txt");

	EXPECT_NE(text.find("add_library(LiteNNCore"), std::string::npos);
	EXPECT_NE(text.find("add_library(LiteNNImporters"), std::string::npos);
	EXPECT_NE(text.find("add_library(LiteNNCUDARuntime"), std::string::npos);
	EXPECT_NE(text.find("add_library(LiteNN INTERFACE)"), std::string::npos);
	EXPECT_NE(text.find("option(LITENN_BUILD_EXAMPLES \"Build LiteNN example programs\" OFF)"), std::string::npos);
	EXPECT_NE(text.find("option(LITENN_BUILD_TOOLS \"Build LiteNN standalone tool programs\" OFF)"), std::string::npos);
	EXPECT_NE(text.find("add_library(LiteNN::LiteNNCore ALIAS LiteNNCore)"), std::string::npos);
	EXPECT_NE(text.find("add_library(LiteNN::LiteNNImporters ALIAS LiteNNImporters)"), std::string::npos);
	EXPECT_NE(text.find("add_library(LiteNN::LiteNNCUDARuntime ALIAS LiteNNCUDARuntime)"), std::string::npos);
	EXPECT_NE(text.find("target_link_libraries(LiteNNImporters PUBLIC LiteNNCore)"), std::string::npos);
	EXPECT_NE(text.find("target_link_libraries(LiteNN INTERFACE LiteNNCore LiteNNImporters)"), std::string::npos);
	EXPECT_NE(text.find("target_link_libraries(LiteNNCUDARuntime PUBLIC LiteNNCore CUDA::cudart CUDA::cublas)"),
	          std::string::npos);
	EXPECT_NE(text.find("target_link_libraries(LiteNN INTERFACE LiteNNCUDARuntime)"), std::string::npos);
	EXPECT_EQ(text.find("target_link_libraries(LiteNNCore PUBLIC CUDA::"), std::string::npos);
	EXPECT_NE(text.find("third_party/simdjson/src/simdjson.cpp"), std::string::npos);

	const auto compilerCmake = ReadSourceFile("src/LiteNN/Compiler/CMakeLists.txt");
	EXPECT_NE(compilerCmake.find("add_library(LiteNNTrainingAOT"), std::string::npos);
	EXPECT_NE(compilerCmake.find("add_library(LiteNN::LiteNNTrainingAOT ALIAS LiteNNTrainingAOT)"),
	          std::string::npos);
	EXPECT_NE(compilerCmake.find("target_link_libraries(LiteNNTrainingAOT PUBLIC LiteNNCompiler)"),
	          std::string::npos);
	EXPECT_NE(compilerCmake.find("src/LiteNN/Training/TrainStepAOTRunner.cpp"), std::string::npos);
	EXPECT_EQ(compilerCmake.find("list(APPEND LITENN_COMPILER_SOURCES\n"
	                             "    ${CMAKE_SOURCE_DIR}/src/LiteNN/Training/TrainStepAOTRunner.cpp"),
	          std::string::npos);

	const auto ggufToolCmake = ReadSourceFile("tools/gguf/CMakeLists.txt");
	const auto torchToolCmake = ReadSourceFile("tools/torch/CMakeLists.txt");
	EXPECT_NE(ggufToolCmake.find("COMPONENT LiteNNTools"), std::string::npos);
	EXPECT_NE(torchToolCmake.find("COMPONENT LiteNNTools"), std::string::npos);
}

TEST(G14PublicApiGuard, UmbrellaHeadersExposeNarrowDeploymentSurfaces)
{
	const auto umbrella = ReadSourceFile("src/LiteNN.h");
	EXPECT_NE(umbrella.find("#include <LiteNNCore.h>"), std::string::npos);
	EXPECT_EQ(umbrella.find("#include <LiteNNImporters.h>"), std::string::npos);
	EXPECT_EQ(umbrella.find("#include <LiteNN/Serialization/ModelIO.h>"), std::string::npos);
	EXPECT_EQ(umbrella.find("#include <LiteNN/Serialization/ModelPackageIO.h>"), std::string::npos);
	EXPECT_EQ(umbrella.find("#include <LiteNN/Serialization/Safetensors.h>"), std::string::npos);
	EXPECT_EQ(umbrella.find("#include <LiteNN/Serialization/TorchManifest.h>"), std::string::npos);

	const auto core = ReadSourceFile("src/LiteNNCore.h");
	EXPECT_NE(core.find("#include <LiteNN/ExecutablePlan.h>"), std::string::npos);
	EXPECT_NE(core.find("#include <LiteNN/Runtime/Scheduler.h>"), std::string::npos);
	EXPECT_EQ(core.find("#include <LiteNN/Serialization/ModelIO.h>"), std::string::npos);
	EXPECT_EQ(core.find("#include <LiteNN/Serialization/ModelPackageIO.h>"), std::string::npos);
	EXPECT_EQ(core.find("#include <LiteNN/Serialization/Safetensors.h>"), std::string::npos);
	EXPECT_EQ(core.find("#include <LiteNN/Serialization/TorchManifest.h>"), std::string::npos);

	const auto importers = ReadSourceFile("src/LiteNNImporters.h");
	EXPECT_NE(importers.find("#include <LiteNNCore.h>"), std::string::npos);
	EXPECT_NE(importers.find("#include <LiteNN/Serialization/ModelPackageIO.h>"), std::string::npos);
	EXPECT_NE(importers.find("#include <LiteNN/Serialization/Safetensors.h>"), std::string::npos);
	EXPECT_NE(importers.find("#include <LiteNN/Serialization/TorchManifest.h>"), std::string::npos);
	EXPECT_EQ(importers.find("#include <LiteNN/Serialization/ModelIO.h>"), std::string::npos);

	const auto packageHeader = ReadSourceFile("src/LiteNN/Serialization/ModelPackageIO.h");
	EXPECT_NE(packageHeader.find("#include <LiteNN/Serialization/ExternalWeights.h>"), std::string::npos);
	EXPECT_EQ(packageHeader.find("#include <LiteNN/Serialization/ModelIO.h>"), std::string::npos);

	const auto compiler = ReadSourceFile("src/LiteNNCompiler.h");
	EXPECT_NE(compiler.find("#include <LiteNNCore.h>"), std::string::npos);
	EXPECT_NE(compiler.find("#include <LiteNN/Compiler/CompiledModule.h>"), std::string::npos);
	EXPECT_NE(compiler.find("#include <LiteNN/Compiler/Dump.h>"), std::string::npos);
	EXPECT_NE(compiler.find("#include <LiteNN/Compiler/Translation/GraphToMLIR.h>"), std::string::npos);
	EXPECT_EQ(compiler.find("#include <LiteNNImporters.h>"), std::string::npos);
}

TEST(G14PublicApiGuard, RawGraphMutationPassesAreInternalDetailScoped)
{
	const auto passHeader = ReadSourceFile("src/LiteNN/Pass.h");
	EXPECT_EQ(passHeader.find("\n\tstruct Pass"), std::string::npos);
	EXPECT_EQ(passHeader.find("namespace Migration"), std::string::npos);
	EXPECT_NE(passHeader.find("struct GraphMutationPass"), std::string::npos);
	EXPECT_NE(passHeader.find("std::span<Detail::GraphMutationPass* const>"), std::string::npos);

	const std::vector<std::string_view> graphMutationPassHeaders{
		"src/LiteNN/Pass/AutogradPass.h",   "src/LiteNN/Pass/ConstFoldPass.h",
		"src/LiteNN/Pass/EGraphPass.h",     "src/LiteNN/Pass/ForwardOnlyPass.h",
		"src/LiteNN/Pass/FusionPass.h",     "src/LiteNN/Pass/InlinePass.h",
	};
	for (const auto header : graphMutationPassHeaders)
	{
		const auto text = ReadSourceFile(header);
		EXPECT_NE(text.find("Detail::GraphMutationPass"), std::string::npos) << header;
		EXPECT_EQ(text.find("public Pass"), std::string::npos) << header;
	}
}

TEST(G14PublicApiGuard, ModelGraphRawGraphAccessIsExplicitlyUnsafe)
{
	const auto executablePlan = ReadSourceFile("src/LiteNN/ExecutablePlan.h");
	const auto modelBuilder = ReadSourceFile("src/LiteNN/ModelBuilder.h");

	EXPECT_NE(executablePlan.find("Graph& UnsafeMutableGraph()"), std::string::npos);
	EXPECT_NE(executablePlan.find("const Graph& UnsafeGraphView()"), std::string::npos);
	EXPECT_NE(executablePlan.find("Graph UnsafeTakeGraph()"), std::string::npos);
	EXPECT_NE(modelBuilder.find("Graph& UnsafeMutableGraph()"), std::string::npos);
	EXPECT_NE(modelBuilder.find("const Graph& UnsafeGraphView()"), std::string::npos);
	EXPECT_NE(modelBuilder.find("Graph UnsafeTakeGraph()"), std::string::npos);
	EXPECT_NE(modelBuilder.find("ExecutablePlan BuildExecutablePlan("), std::string::npos);

	EXPECT_EQ(executablePlan.find("Graph& MutableGraph()"), std::string::npos);
	EXPECT_EQ(executablePlan.find("const Graph& GraphView()"), std::string::npos);
	EXPECT_EQ(executablePlan.find("Graph TakeGraph()"), std::string::npos);
	EXPECT_EQ(modelBuilder.find("Graph& MutableGraph()"), std::string::npos);
	EXPECT_EQ(modelBuilder.find("const Graph& GraphView()"), std::string::npos);
	EXPECT_EQ(modelBuilder.find("Graph TakeGraph()"), std::string::npos);
}

TEST(G14PublicApiGuard, TensorRawDataCompatibilityForwarderIsRemoved)
{
	const auto tensorHeader = ReadSourceFile("src/LiteNN/Tensor.h");
	EXPECT_NE(tensorHeader.find("UnsafeRawData()"), std::string::npos);
	EXPECT_EQ(tensorHeader.find(" RawData()"), std::string::npos);
	EXPECT_EQ(tensorHeader.find(".RawData()"), std::string::npos);
	EXPECT_EQ(tensorHeader.find("->RawData()"), std::string::npos);
}
