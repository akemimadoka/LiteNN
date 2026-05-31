#ifndef LITENN_OP_SCHEMA_H
#define LITENN_OP_SCHEMA_H

#include <LiteNN/Graph.h>
#include <LiteNN/TensorType.h>
#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace LiteNN
{
	enum class OpCategory
	{
		Source,
		Elementwise,
		LinearAlgebra,
		Reduction,
		Shape,
		DataMovement,
		ControlFlow,
		State,
		Optimizer,
		NeuralNetwork,
		Custom
	};

	enum class OpEffect
	{
		Pure,
		ReadsGraphState,
		WritesRuntimeState,
		ControlFlow
	};

	enum class BackendSupportLevel
	{
		Unsupported,
		Fallback,
		Native
	};

	inline constexpr std::string_view BackendCPUInterpreter = "CPUInterpreter";
	inline constexpr std::string_view BackendCPUAOT = "CPUAOT";
	inline constexpr std::string_view BackendCUDANative = "CUDANative";
	inline constexpr std::string_view BackendCUDABridge = "CUDABridge";
	inline constexpr std::string_view BackendMobile = "Mobile";

	inline constexpr std::array<std::string_view, 5> DefaultBackendNames{
		BackendCPUInterpreter, BackendCPUAOT, BackendCUDANative, BackendCUDABridge, BackendMobile
	};

	struct BackendCapability
	{
		std::string backend;
		BackendSupportLevel support{ BackendSupportLevel::Unsupported };
		std::vector<DataType> dtypes;
		std::vector<TensorLayoutKind> layouts;
		std::vector<TensorMemorySpace> memorySpaces;
		OpEffect memoryEffect{ OpEffect::Pure };
		std::string lowering;
		std::string fallback;
		double relativeCost{ 1.0 };
	};

	struct OpCoverageRow
	{
		std::string kind;
		OpCategory category{ OpCategory::Custom };
		std::vector<BackendCapability> capabilities;
	};

	struct OpSchema
	{
		static constexpr std::size_t DynamicArity = std::numeric_limits<std::size_t>::max();

		std::string kind;
		OpCategory category{ OpCategory::Custom };
		OpEffect effect{ OpEffect::Pure };
		std::size_t minInputs{};
		std::size_t maxInputs{};
		std::size_t minOutputs{ 1 };
		std::size_t maxOutputs{ 1 };
		bool hasShapeInference{ false };
		bool hasVerifier{ false };
		std::vector<BackendCapability> capabilities;

		bool AllowsInputCount(std::size_t count) const noexcept
		{
			return count >= minInputs && (maxInputs == DynamicArity || count <= maxInputs);
		}

		bool AllowsOutputCount(std::size_t count) const noexcept
		{
			return count >= minOutputs && (maxOutputs == DynamicArity || count <= maxOutputs);
		}

		const BackendCapability* FindCapability(std::string_view backend) const
		{
			const auto it = std::ranges::find_if(capabilities, [&](const BackendCapability& capability) {
				return std::string_view{ capability.backend } == backend;
			});
			return it == capabilities.end() ? nullptr : &*it;
		}

		bool SupportsBackend(std::string_view backend) const
		{
			const auto* capability = FindCapability(backend);
			return capability && capability->support != BackendSupportLevel::Unsupported;
		}
	};

	class OpSchemaRegistry
	{
	public:
		void Register(OpSchema schema)
		{
			const auto [it, inserted] = indexByKind_.try_emplace(schema.kind, schemas_.size());
			if (!inserted)
			{
				schemas_[it->second] = std::move(schema);
				return;
			}
			schemas_.push_back(std::move(schema));
		}

		void RegisterCapability(std::string_view kind, BackendCapability capability)
		{
			auto& schema = MutableRequire(kind);
			const auto it = std::ranges::find_if(schema.capabilities, [&](const BackendCapability& existing) {
				return existing.backend == capability.backend;
			});
			if (it == schema.capabilities.end())
			{
				schema.capabilities.push_back(std::move(capability));
				return;
			}
			*it = std::move(capability);
		}

		const OpSchema* Find(std::string_view kind) const
		{
			const auto it = indexByKind_.find(std::string(kind));
			if (it == indexByKind_.end())
			{
				return nullptr;
			}
			return &schemas_[it->second];
		}

		const OpSchema& Require(std::string_view kind) const
		{
			if (const auto* schema = Find(kind))
			{
				return *schema;
			}
			throw std::runtime_error("Unknown LiteNN op schema: " + std::string(kind));
		}

		OpSchema& MutableRequire(std::string_view kind)
		{
			const auto it = indexByKind_.find(std::string(kind));
			if (it == indexByKind_.end())
			{
				throw std::runtime_error("Unknown LiteNN op schema: " + std::string(kind));
			}
			return schemas_[it->second];
		}

		bool Contains(std::string_view kind) const
		{
			return Find(kind) != nullptr;
		}

		std::span<const OpSchema> Schemas() const noexcept
		{
			return schemas_;
		}

		std::vector<OpCoverageRow> CoverageReport() const
		{
			return CoverageReport(std::span<const std::string_view>{ DefaultBackendNames });
		}

		std::vector<OpCoverageRow> CoverageReport(std::span<const std::string_view> backends) const
		{
			std::vector<OpCoverageRow> rows;
			rows.reserve(schemas_.size());
			for (const auto& schema : schemas_)
			{
				OpCoverageRow row{ .kind = schema.kind, .category = schema.category };
				row.capabilities.reserve(backends.size());
				for (const auto backend : backends)
				{
					if (const auto* capability = schema.FindCapability(backend))
					{
						row.capabilities.push_back(*capability);
					}
					else
					{
						row.capabilities.push_back({ .backend = std::string(backend) });
					}
				}
				rows.push_back(std::move(row));
			}
			return rows;
		}

	private:
		std::vector<OpSchema> schemas_;
		std::unordered_map<std::string, std::size_t> indexByKind_;
	};

	namespace Detail
	{
		template <typename T>
		constexpr std::string_view RawTypeName()
		{
#if defined(__clang__) || defined(__GNUC__)
			constexpr std::string_view function = __PRETTY_FUNCTION__;
			constexpr std::string_view key = "T = ";
			const auto start = function.find(key) + key.size();
			const auto end = function.find_first_of(";]", start);
			return function.substr(start, end - start);
#elif defined(_MSC_VER)
			constexpr std::string_view function = __FUNCSIG__;
			constexpr std::string_view key = "RawTypeName<";
			const auto start = function.find(key) + key.size();
			const auto end = function.find(">(void)", start);
			return function.substr(start, end - start);
#else
			return "UnknownNode";
#endif
		}

		constexpr std::string_view StripNamespace(std::string_view name)
		{
			const auto pos = name.rfind("::");
			if (pos == std::string_view::npos)
			{
				return name;
			}
			return name.substr(pos + 2);
		}

		template <typename T>
		constexpr std::string_view NodeTypeName()
		{
			return StripNamespace(RawTypeName<T>());
		}

		template <typename T>
		struct NodeSchemaTraits
		{
			static constexpr OpCategory Category = OpCategory::Custom;
			static constexpr OpEffect Effect = OpEffect::Pure;
			static constexpr std::size_t MinInputs = 0;
			static constexpr std::size_t MaxInputs = OpSchema::DynamicArity;
			static constexpr std::size_t MinOutputs = 1;
			static constexpr std::size_t MaxOutputs = OpSchema::DynamicArity;
			static constexpr bool HasShapeInference = false;
			static constexpr bool HasVerifier = false;
		};

		template <>
		struct NodeSchemaTraits<ParamRefNode>
		{
			static constexpr OpCategory Category = OpCategory::Source;
			static constexpr OpEffect Effect = OpEffect::Pure;
			static constexpr std::size_t MinInputs = 0;
			static constexpr std::size_t MaxInputs = 0;
			static constexpr std::size_t MinOutputs = 1;
			static constexpr std::size_t MaxOutputs = 1;
			static constexpr bool HasShapeInference = true;
			static constexpr bool HasVerifier = true;
		};

		template <>
		struct NodeSchemaTraits<ConstantNode> : NodeSchemaTraits<ParamRefNode>
		{
		};

		template <>
		struct NodeSchemaTraits<QuantizedConstantNode> : NodeSchemaTraits<ParamRefNode>
		{
		};

		template <>
		struct NodeSchemaTraits<VariableRefNode> : NodeSchemaTraits<ParamRefNode>
		{
			static constexpr OpEffect Effect = OpEffect::ReadsGraphState;
		};

		template <>
		struct NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::Elementwise;
			static constexpr OpEffect Effect = OpEffect::Pure;
			static constexpr std::size_t MinInputs = 1;
			static constexpr std::size_t MaxInputs = 1;
			static constexpr std::size_t MinOutputs = 1;
			static constexpr std::size_t MaxOutputs = 1;
			static constexpr bool HasShapeInference = true;
			static constexpr bool HasVerifier = true;
		};

		template <>
		struct NodeSchemaTraits<BinaryOpNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr std::size_t MinInputs = 2;
			static constexpr std::size_t MaxInputs = 2;
		};

		template <>
		struct NodeSchemaTraits<ReduceOpNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::Reduction;
		};

		template <>
		struct NodeSchemaTraits<CastNode> : NodeSchemaTraits<UnaryOpNode>
		{
		};

		template <>
		struct NodeSchemaTraits<QuantizeNode> : NodeSchemaTraits<UnaryOpNode>
		{
		};

		template <>
		struct NodeSchemaTraits<DequantizeNode> : NodeSchemaTraits<UnaryOpNode>
		{
		};

		template <>
		struct NodeSchemaTraits<CallNode>
		{
			static constexpr OpCategory Category = OpCategory::ControlFlow;
			static constexpr OpEffect Effect = OpEffect::ControlFlow;
			static constexpr std::size_t MinInputs = 0;
			static constexpr std::size_t MaxInputs = OpSchema::DynamicArity;
			static constexpr std::size_t MinOutputs = 0;
			static constexpr std::size_t MaxOutputs = OpSchema::DynamicArity;
			static constexpr bool HasShapeInference = true;
			static constexpr bool HasVerifier = true;
		};

		template <>
		struct NodeSchemaTraits<CondNode> : NodeSchemaTraits<CallNode>
		{
			static constexpr std::size_t MinInputs = 1;
		};

		template <>
		struct NodeSchemaTraits<WhileNode> : NodeSchemaTraits<CallNode>
		{
		};

		template <>
		struct NodeSchemaTraits<SaveActivationNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::State;
			static constexpr OpEffect Effect = OpEffect::WritesRuntimeState;
		};

		template <>
		struct NodeSchemaTraits<LoadActivationNode> : NodeSchemaTraits<ParamRefNode>
		{
			static constexpr OpCategory Category = OpCategory::State;
			static constexpr OpEffect Effect = OpEffect::ReadsGraphState;
		};

		template <>
		struct NodeSchemaTraits<TapeSaveActivationNode> : NodeSchemaTraits<SaveActivationNode>
		{
		};

		template <>
		struct NodeSchemaTraits<TapeLoadActivationNode> : NodeSchemaTraits<LoadActivationNode>
		{
		};

		template <>
		struct NodeSchemaTraits<ReshapeNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::Shape;
		};

		template <>
		struct NodeSchemaTraits<PermuteNode> : NodeSchemaTraits<ReshapeNode>
		{
		};

		template <>
		struct NodeSchemaTraits<BroadcastToNode> : NodeSchemaTraits<ReshapeNode>
		{
		};

		template <>
		struct NodeSchemaTraits<PadNode> : NodeSchemaTraits<ReshapeNode>
		{
		};

		template <>
		struct NodeSchemaTraits<GatherNode> : NodeSchemaTraits<BinaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::DataMovement;
		};

		template <>
		struct NodeSchemaTraits<ScatterNode> : NodeSchemaTraits<GatherNode>
		{
			static constexpr std::size_t MinInputs = 3;
			static constexpr std::size_t MaxInputs = 3;
		};

		template <>
		struct NodeSchemaTraits<ScanNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::DataMovement;
		};

		template <>
		struct NodeSchemaTraits<SSMScanNode> : NodeSchemaTraits<ScanNode>
		{
			static constexpr std::size_t MinInputs = 5;
			static constexpr std::size_t MaxInputs = 6;
		};

		template <>
		struct NodeSchemaTraits<RWKVWKVNode> : NodeSchemaTraits<ScanNode>
		{
			static constexpr std::size_t MinInputs = 5;
			static constexpr std::size_t MaxInputs = 5;
		};

		template <>
		struct NodeSchemaTraits<SoftmaxNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::NeuralNetwork;
		};

		template <>
		struct NodeSchemaTraits<CrossEntropyLossNode> : NodeSchemaTraits<BinaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::NeuralNetwork;
		};

		template <>
		struct NodeSchemaTraits<CrossEntropyLossBackwardNode> : NodeSchemaTraits<ScatterNode>
		{
			static constexpr OpCategory Category = OpCategory::NeuralNetwork;
		};

		template <>
		struct NodeSchemaTraits<NormalizationNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::NeuralNetwork;
			static constexpr std::size_t MaxInputs = 3;
		};

		template <>
		struct NodeSchemaTraits<BatchMatMulNode> : NodeSchemaTraits<BinaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::LinearAlgebra;
		};

		template <>
		struct NodeSchemaTraits<OutProdNode> : NodeSchemaTraits<BatchMatMulNode>
		{
		};

		template <>
		struct NodeSchemaTraits<TimestepEmbeddingNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::NeuralNetwork;
		};

		template <>
		struct NodeSchemaTraits<SolveTriNode> : NodeSchemaTraits<BatchMatMulNode>
		{
		};

		template <>
		struct NodeSchemaTraits<SGDStepNode> : NodeSchemaTraits<ScatterNode>
		{
			static constexpr OpCategory Category = OpCategory::Optimizer;
			static constexpr std::size_t MinInputs = 2;
			static constexpr std::size_t MaxInputs = 3;
			static constexpr std::size_t MinOutputs = 1;
			static constexpr std::size_t MaxOutputs = 2;
		};

		template <>
		struct NodeSchemaTraits<AdamWStepNode>
		{
			static constexpr OpCategory Category = OpCategory::Optimizer;
			static constexpr OpEffect Effect = OpEffect::Pure;
			static constexpr std::size_t MinInputs = 4;
			static constexpr std::size_t MaxInputs = 4;
			static constexpr std::size_t MinOutputs = 3;
			static constexpr std::size_t MaxOutputs = 3;
			static constexpr bool HasShapeInference = true;
			static constexpr bool HasVerifier = true;
		};

		template <>
		struct NodeSchemaTraits<Im2ColNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::DataMovement;
		};

		template <>
		struct NodeSchemaTraits<Conv2DNode> : NodeSchemaTraits<NormalizationNode>
		{
			static constexpr OpCategory Category = OpCategory::NeuralNetwork;
			static constexpr std::size_t MinInputs = 2;
			static constexpr std::size_t MaxInputs = 3;
		};

		template <>
		struct NodeSchemaTraits<ConvTranspose2DNode> : NodeSchemaTraits<Conv2DNode>
		{
		};

		template <>
		struct NodeSchemaTraits<Pool2DNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::NeuralNetwork;
		};

		template <>
		struct NodeSchemaTraits<UpsampleNode> : NodeSchemaTraits<Pool2DNode>
		{
		};

		template <>
		struct NodeSchemaTraits<ConcatNode> : NodeSchemaTraits<CallNode>
		{
			static constexpr OpCategory Category = OpCategory::DataMovement;
		};

		template <>
		struct NodeSchemaTraits<SliceNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::DataMovement;
		};

		template <>
		struct NodeSchemaTraits<GetRowsNode> : NodeSchemaTraits<BinaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::DataMovement;
		};

		template <>
		struct NodeSchemaTraits<ArgsortNode> : NodeSchemaTraits<UnaryOpNode>
		{
			static constexpr OpCategory Category = OpCategory::DataMovement;
		};

		template <>
		struct NodeSchemaTraits<MulMatIdNode> : NodeSchemaTraits<ScatterNode>
		{
			static constexpr OpCategory Category = OpCategory::LinearAlgebra;
		};

		template <>
		struct NodeSchemaTraits<FusedOpNode> : NodeSchemaTraits<CallNode>
		{
			static constexpr OpCategory Category = OpCategory::Custom;
		};

		template <typename NodeT>
		OpSchema MakeNodeSchema()
		{
			using Traits = NodeSchemaTraits<NodeT>;
			return {
				.kind = std::string(NodeTypeName<NodeT>()),
				.category = Traits::Category,
				.effect = Traits::Effect,
				.minInputs = Traits::MinInputs,
				.maxInputs = Traits::MaxInputs,
				.minOutputs = Traits::MinOutputs,
				.maxOutputs = Traits::MaxOutputs,
				.hasShapeInference = Traits::HasShapeInference,
				.hasVerifier = Traits::HasVerifier,
			};
		}

		template <typename Variant, std::size_t... Indices>
		void RegisterVariantSchemas(OpSchemaRegistry& registry, std::index_sequence<Indices...>)
		{
			(registry.Register(MakeNodeSchema<std::variant_alternative_t<Indices, Variant>>()), ...);
		}
	} // namespace Detail

	template <typename NodeT>
	constexpr std::string_view OpKindName()
	{
		return Detail::NodeTypeName<NodeT>();
	}

	inline std::string OpKindName(const NodeVariant& node)
	{
		return std::visit([]<typename NodeT>(const NodeT&) { return std::string(OpKindName<NodeT>()); }, node);
	}

	inline OpSchemaRegistry BuildDefaultOpSchemaRegistry()
	{
		OpSchemaRegistry registry;
		Detail::RegisterVariantSchemas<NodeVariant>(registry,
		                                            std::make_index_sequence<std::variant_size_v<NodeVariant>>{});
		for (const auto& schema : registry.Schemas())
		{
			registry.RegisterCapability(schema.kind,
			                            { .backend = std::string(BackendCPUInterpreter),
			                              .support = BackendSupportLevel::Native,
			                              .layouts = { TensorLayoutKind::RowMajor },
			                              .memorySpaces = { TensorMemorySpace::Host },
			                              .memoryEffect = schema.effect,
			                              .lowering = "Runtime::Interpreter",
			                              .relativeCost = 1.0 });
			for (const auto backend :
			     { BackendCPUAOT, BackendCUDANative, BackendCUDABridge, BackendMobile })
			{
				registry.RegisterCapability(schema.kind,
				                            { .backend = std::string(backend),
				                              .support = BackendSupportLevel::Unsupported,
				                              .memoryEffect = schema.effect,
				                              .fallback = std::string(BackendCPUInterpreter) });
			}
		}
		return registry;
	}

	inline const OpSchemaRegistry& DefaultOpSchemaRegistry()
	{
		static const auto registry = BuildDefaultOpSchemaRegistry();
		return registry;
	}

	inline std::vector<NodeOutput> NodeInputs(const NodeVariant& node)
	{
		return std::visit(
		    [](const auto& value) -> std::vector<NodeOutput> {
			    using T = std::decay_t<decltype(value)>;
			    if constexpr (std::same_as<T, UnaryOpNode> || std::same_as<T, CastNode> ||
			                  std::same_as<T, QuantizeNode> || std::same_as<T, DequantizeNode> ||
			                  std::same_as<T, SaveActivationNode> || std::same_as<T, TapeSaveActivationNode> ||
			                  std::same_as<T, ReduceOpNode> || std::same_as<T, ReshapeNode> ||
			                  std::same_as<T, PermuteNode> || std::same_as<T, BroadcastToNode> ||
			                  std::same_as<T, PadNode> || std::same_as<T, ScanNode> ||
			                  std::same_as<T, SoftmaxNode> || std::same_as<T, Im2ColNode> || std::same_as<T, Pool2DNode> ||
			                  std::same_as<T, UpsampleNode> || std::same_as<T, SliceNode> ||
			                  std::same_as<T, ArgsortNode>)
			    {
				    return { value.input };
			    }
			    else if constexpr (std::same_as<T, TimestepEmbeddingNode>)
			    {
				    return { value.timesteps };
			    }
			    else if constexpr (std::same_as<T, BinaryOpNode> || std::same_as<T, BatchMatMulNode> ||
			                       std::same_as<T, OutProdNode>)
			    {
				    return { value.lhs, value.rhs };
			    }
			    else if constexpr (std::same_as<T, CallNode>)
			    {
				    return value.args;
			    }
			    else if constexpr (std::same_as<T, CondNode>)
			    {
				    auto inputs = value.args;
				    inputs.insert(inputs.begin(), value.condition);
				    return inputs;
			    }
			    else if constexpr (std::same_as<T, WhileNode>)
			    {
				    return value.initArgs;
			    }
			    else if constexpr (std::same_as<T, GatherNode>)
			    {
				    return { value.data, value.indices };
			    }
			    else if constexpr (std::same_as<T, ScatterNode>)
			    {
				    return { value.data, value.indices, value.updates };
			    }
			    else if constexpr (std::same_as<T, SSMScanNode>)
			    {
				    std::vector<NodeOutput> inputs{ value.state, value.dt, value.a, value.b, value.c };
				    if (value.d)
				    {
					    inputs.push_back(*value.d);
				    }
				    return inputs;
			    }
			    else if constexpr (std::same_as<T, RWKVWKVNode>)
			    {
				    return { value.key, value.value, value.receptance, value.timeDecay, value.timeFirst };
			    }
			    else if constexpr (std::same_as<T, CrossEntropyLossNode>)
			    {
				    return { value.logits, value.labels };
			    }
			    else if constexpr (std::same_as<T, CrossEntropyLossBackwardNode>)
			    {
				    return { value.grad, value.logits, value.labels };
			    }
			    else if constexpr (std::same_as<T, NormalizationNode>)
			    {
				    std::vector<NodeOutput> inputs{ value.input };
				    if (value.scale)
				    {
					    inputs.push_back(*value.scale);
				    }
				    if (value.bias)
				    {
					    inputs.push_back(*value.bias);
				    }
				    return inputs;
			    }
			    else if constexpr (std::same_as<T, SolveTriNode>)
			    {
				    return { value.a, value.b };
			    }
			    else if constexpr (std::same_as<T, SGDStepNode>)
			    {
				    std::vector<NodeOutput> inputs{ value.parameter, value.gradient };
				    if (value.velocity)
				    {
					    inputs.push_back(*value.velocity);
				    }
				    return inputs;
			    }
			    else if constexpr (std::same_as<T, AdamWStepNode>)
			    {
				    return { value.parameter, value.gradient, value.firstMoment, value.secondMoment };
			    }
			    else if constexpr (std::same_as<T, Conv2DNode> || std::same_as<T, ConvTranspose2DNode>)
			    {
				    std::vector<NodeOutput> inputs{ value.input, value.weight };
				    if (value.bias)
				    {
					    inputs.push_back(*value.bias);
				    }
				    return inputs;
			    }
			    else if constexpr (std::same_as<T, ConcatNode>)
			    {
				    return value.inputs;
			    }
			    else if constexpr (std::same_as<T, GetRowsNode>)
			    {
				    return { value.data, value.indices };
			    }
			    else if constexpr (std::same_as<T, MulMatIdNode>)
			    {
				    return { value.as, value.b, value.ids };
			    }
			    else if constexpr (std::same_as<T, FusedOpNode>)
			    {
				    return value.args;
			    }
			    else
			    {
				    return {};
			    }
		    },
		    node);
	}
} // namespace LiteNN

#endif
