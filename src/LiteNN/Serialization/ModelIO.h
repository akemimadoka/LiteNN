#ifndef LITENN_SERIALIZATION_MODELIO_H
#define LITENN_SERIALIZATION_MODELIO_H

#include <LiteNN/Graph.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace LiteNN::Serialization
{
	namespace Detail
	{
		constexpr std::array<char, 8> kModelMagic = { 'L', 'T', 'N', 'N', 'M', 'D', 'L', '\0' };
		constexpr std::uint32_t kModelVersion = 2;

		enum class NodeKind : std::uint32_t
		{
			ParamRef = 0,
			Constant,
			VariableRef,
			UnaryOp,
			BinaryOp,
			Call,
			Cast,
			Cond,
			While,
			SaveActivation,
			LoadActivation,
			TapeSaveActivation,
			TapeLoadActivation,
			ReduceOp,
			Reshape,
			Concat,
			Slice,
			FusedOp,
		};

		inline std::size_t ElementByteSize(DataType dtype)
		{
			switch (dtype)
			{
			case DataType::Float32:
				return sizeof(float);
			case DataType::Float64:
				return sizeof(double);
			case DataType::Int32:
				return sizeof(std::int32_t);
			case DataType::Int64:
				return sizeof(std::int64_t);
			case DataType::Bool:
				return sizeof(bool);
			}
			throw std::runtime_error("Invalid data type");
		}

		inline void EnsureWrite(const std::ostream& out)
		{
			if (!out)
			{
				throw std::runtime_error("Failed to write LiteNN model");
			}
		}

		inline void EnsureRead(const std::istream& in)
		{
			if (!in)
			{
				throw std::runtime_error("LiteNN model is truncated or unreadable");
			}
		}

		template <typename T>
		void WriteScalar(std::ostream& out, T value)
		{
			out.write(reinterpret_cast<const char*>(&value), sizeof(T));
			EnsureWrite(out);
		}

		template <typename T>
		T ReadScalar(std::istream& in)
		{
			T value{};
			in.read(reinterpret_cast<char*>(&value), sizeof(T));
			EnsureRead(in);
			return value;
		}

		inline void WriteSize(std::ostream& out, std::size_t value)
		{
			WriteScalar(out, static_cast<std::uint64_t>(value));
		}

		inline std::size_t ReadSize(std::istream& in)
		{
			return static_cast<std::size_t>(ReadScalar<std::uint64_t>(in));
		}

		inline void WriteDataType(std::ostream& out, DataType dtype)
		{
			WriteScalar(out, static_cast<std::uint32_t>(dtype));
		}

		inline DataType ReadDataType(std::istream& in)
		{
			return static_cast<DataType>(ReadScalar<std::uint32_t>(in));
		}

		inline void WriteShape(std::ostream& out, std::span<const std::size_t> shape)
		{
			WriteSize(out, shape.size());
			for (const auto dim : shape)
			{
				WriteSize(out, dim);
			}
		}

		inline std::vector<std::size_t> ReadShape(std::istream& in)
		{
			std::vector<std::size_t> shape(ReadSize(in));
			for (auto& dim : shape)
			{
				dim = ReadSize(in);
			}
			return shape;
		}

		inline void WriteString(std::ostream& out, std::string_view value)
		{
			WriteSize(out, value.size());
			out.write(value.data(), static_cast<std::streamsize>(value.size()));
			EnsureWrite(out);
		}

		inline std::string ReadString(std::istream& in)
		{
			std::string value(ReadSize(in), '\0');
			in.read(value.data(), static_cast<std::streamsize>(value.size()));
			EnsureRead(in);
			return value;
		}

		inline void WriteStringList(std::ostream& out, std::span<const std::string> values)
		{
			WriteSize(out, values.size());
			for (const auto& value : values)
			{
				WriteString(out, value);
			}
		}

		inline std::vector<std::string> ReadStringList(std::istream& in)
		{
			std::vector<std::string> values(ReadSize(in));
			for (auto& value : values)
			{
				value = ReadString(in);
			}
			return values;
		}

		inline void WriteNodeOutput(std::ostream& out, NodeOutput output)
		{
			WriteSize(out, output.node);
			WriteSize(out, output.port);
		}

		inline NodeOutput ReadNodeOutput(std::istream& in)
		{
			return { ReadSize(in), ReadSize(in) };
		}

		inline void WriteNodeOutputList(std::ostream& out, std::span<const NodeOutput> outputs)
		{
			WriteSize(out, outputs.size());
			for (const auto output : outputs)
			{
				WriteNodeOutput(out, output);
			}
		}

		inline std::vector<NodeOutput> ReadNodeOutputList(std::istream& in)
		{
			std::vector<NodeOutput> outputs(ReadSize(in));
			for (auto& output : outputs)
			{
				output = ReadNodeOutput(in);
			}
			return outputs;
		}

		inline void WriteOutputInfo(std::ostream& out, const OutputInfo& info)
		{
			WriteDataType(out, info.dtype);
			WriteShape(out, info.shape);
		}

		inline OutputInfo ReadOutputInfo(std::istream& in)
		{
			return { ReadDataType(in), ReadShape(in) };
		}

		inline void WriteOutputInfoList(std::ostream& out, std::span<const OutputInfo> infos)
		{
			WriteSize(out, infos.size());
			for (const auto& info : infos)
			{
				WriteOutputInfo(out, info);
			}
		}

		inline std::vector<OutputInfo> ReadOutputInfoList(std::istream& in)
		{
			std::vector<OutputInfo> infos(ReadSize(in));
			for (auto& info : infos)
			{
				info = ReadOutputInfo(in);
			}
			return infos;
		}

		template <Device D>
		void WriteTensor(std::ostream& out, const Tensor<D>& tensor)
		{
			auto cpuTensor = tensor.CopyToDevice(CPU{});
			WriteDataType(out, cpuTensor.DType());
			WriteShape(out, cpuTensor.Shape().Dims);
			const auto byteCount = cpuTensor.NumElements() * ElementByteSize(cpuTensor.DType());
			out.write(static_cast<const char*>(cpuTensor.RawData()), static_cast<std::streamsize>(byteCount));
			EnsureWrite(out);
		}

		inline Tensor<CPU> ReadTensor(std::istream& in)
		{
			const auto dtype = ReadDataType(in);
			const auto shape = ReadShape(in);
			Tensor<CPU> tensor(Uninitialized, ShapeView{ shape }, dtype, CPU{});
			const auto byteCount = tensor.NumElements() * ElementByteSize(dtype);
			in.read(static_cast<char*>(tensor.RawData()), static_cast<std::streamsize>(byteCount));
			EnsureRead(in);
			return tensor;
		}

		inline void WriteNode(std::ostream& out, const NodeEntry& entry)
		{
			WriteOutputInfoList(out, entry.outputInfos);
			std::visit(
			    [&](const auto& node) {
				    using T = std::decay_t<decltype(node)>;
				    if constexpr (std::same_as<T, ParamRefNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::ParamRef));
					    WriteSize(out, node.paramIndex);
				    }
				    else if constexpr (std::same_as<T, ConstantNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Constant));
					    WriteTensor(out, node.value);
				    }
				    else if constexpr (std::same_as<T, VariableRefNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::VariableRef));
					    WriteSize(out, node.variableIndex);
				    }
				    else if constexpr (std::same_as<T, UnaryOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::UnaryOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.input);
				    }
				    else if constexpr (std::same_as<T, BinaryOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::BinaryOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.lhs);
					    WriteNodeOutput(out, node.rhs);
				    }
				    else if constexpr (std::same_as<T, CallNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Call));
					    WriteSize(out, node.callee);
					    WriteNodeOutputList(out, node.args);
				    }
				    else if constexpr (std::same_as<T, CastNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Cast));
					    WriteNodeOutput(out, node.input);
					    WriteDataType(out, node.targetType);
				    }
				    else if constexpr (std::same_as<T, CondNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Cond));
					    WriteNodeOutput(out, node.condition);
					    WriteSize(out, node.thenBranch);
					    WriteSize(out, node.elseBranch);
					    WriteNodeOutputList(out, node.args);
				    }
				    else if constexpr (std::same_as<T, WhileNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::While));
					    WriteSize(out, node.condBranch);
					    WriteSize(out, node.bodyBranch);
					    WriteNodeOutputList(out, node.initArgs);
				    }
				    else if constexpr (std::same_as<T, SaveActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::SaveActivation));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.slotId);
				    }
				    else if constexpr (std::same_as<T, LoadActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::LoadActivation));
					    WriteSize(out, node.slotId);
				    }
				    else if constexpr (std::same_as<T, TapeSaveActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::TapeSaveActivation));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.tapeSlotId);
				    }
				    else if constexpr (std::same_as<T, TapeLoadActivationNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::TapeLoadActivation));
					    WriteSize(out, node.tapeSlotId);
				    }
				    else if constexpr (std::same_as<T, ReduceOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::ReduceOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.op));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
				    }
				    else if constexpr (std::same_as<T, ReshapeNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Reshape));
					    WriteNodeOutput(out, node.input);
					    WriteShape(out, node.targetShape);
				    }
				    else if constexpr (std::same_as<T, ConcatNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Concat));
					    WriteNodeOutputList(out, node.inputs);
					    WriteSize(out, node.axis);
				    }
				    else if constexpr (std::same_as<T, SliceNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::Slice));
					    WriteNodeOutput(out, node.input);
					    WriteSize(out, node.axis);
					    WriteSize(out, node.start);
					    WriteSize(out, node.length);
				    }
				    else if constexpr (std::same_as<T, FusedOpNode>)
				    {
					    WriteScalar(out, static_cast<std::uint32_t>(NodeKind::FusedOp));
					    WriteScalar(out, static_cast<std::uint32_t>(node.pattern));
					    WriteSize(out, node.body);
					    WriteNodeOutputList(out, node.args);
				    }
			    },
			    entry.node);
		}

		inline NodeVariant ReadNodePayload(std::istream& in)
		{
			const auto kind = static_cast<NodeKind>(ReadScalar<std::uint32_t>(in));
			switch (kind)
			{
			case NodeKind::ParamRef:
				return ParamRefNode{ ReadSize(in) };
			case NodeKind::Constant:
				return ConstantNode{ ReadTensor(in).CopyToDevice(PolymorphicDevice{ CPU{} }) };
			case NodeKind::VariableRef:
				return VariableRefNode{ ReadSize(in) };
			case NodeKind::UnaryOp: {
				const auto op = static_cast<UnaryOp>(ReadScalar<std::uint32_t>(in));
				return UnaryOpNode{ op, ReadNodeOutput(in) };
			}
			case NodeKind::BinaryOp: {
				const auto op = static_cast<BinaryOp>(ReadScalar<std::uint32_t>(in));
				const auto lhs = ReadNodeOutput(in);
				const auto rhs = ReadNodeOutput(in);
				return BinaryOpNode{ op, lhs, rhs };
			}
			case NodeKind::Call: {
				const auto callee = ReadSize(in);
				return CallNode{ callee, ReadNodeOutputList(in) };
			}
			case NodeKind::Cast: {
				const auto input = ReadNodeOutput(in);
				return CastNode{ input, ReadDataType(in) };
			}
			case NodeKind::Cond: {
				const auto condition = ReadNodeOutput(in);
				const auto thenBranch = ReadSize(in);
				const auto elseBranch = ReadSize(in);
				return CondNode{ condition, thenBranch, elseBranch, ReadNodeOutputList(in) };
			}
			case NodeKind::While: {
				const auto condBranch = ReadSize(in);
				const auto bodyBranch = ReadSize(in);
				return WhileNode{ condBranch, bodyBranch, ReadNodeOutputList(in) };
			}
			case NodeKind::SaveActivation: {
				const auto input = ReadNodeOutput(in);
				return SaveActivationNode{ input, ReadSize(in) };
			}
			case NodeKind::LoadActivation:
				return LoadActivationNode{ ReadSize(in) };
			case NodeKind::TapeSaveActivation: {
				const auto input = ReadNodeOutput(in);
				return TapeSaveActivationNode{ input, ReadSize(in) };
			}
			case NodeKind::TapeLoadActivation:
				return TapeLoadActivationNode{ ReadSize(in) };
			case NodeKind::ReduceOp: {
				const auto op = static_cast<ReduceOp>(ReadScalar<std::uint32_t>(in));
				const auto input = ReadNodeOutput(in);
				return ReduceOpNode{ op, input, ReadSize(in) };
			}
			case NodeKind::Reshape: {
				const auto input = ReadNodeOutput(in);
				return ReshapeNode{ input, ReadShape(in) };
			}
			case NodeKind::Concat: {
				auto inputs = ReadNodeOutputList(in);
				return ConcatNode{ std::move(inputs), ReadSize(in) };
			}
			case NodeKind::Slice: {
				const auto input = ReadNodeOutput(in);
				const auto axis = ReadSize(in);
				const auto start = ReadSize(in);
				return SliceNode{ input, axis, start, ReadSize(in) };
			}
			case NodeKind::FusedOp: {
				const auto pattern = static_cast<FusionPattern>(ReadScalar<std::uint32_t>(in));
				const auto body = ReadSize(in);
				return FusedOpNode{ pattern, body, ReadNodeOutputList(in) };
			}
			}
			throw std::runtime_error("LiteNN model contains an unknown node kind");
		}

		inline void WriteSubgraph(std::ostream& out, const Subgraph& subgraph)
		{
			WriteSize(out, subgraph.Params().size());
			for (const auto& param : subgraph.Params())
			{
				WriteDataType(out, param.dtype);
				WriteShape(out, param.shape);
			}

			WriteSize(out, subgraph.NodeCount());
			for (const auto& node : subgraph.Nodes())
			{
				WriteNode(out, node);
			}
			WriteNodeOutputList(out, subgraph.Results());
		}

		inline Subgraph ReadSubgraph(std::istream& in)
		{
			Subgraph subgraph;
			const auto paramCount = ReadSize(in);
			for (std::size_t i = 0; i < paramCount; ++i)
			{
				const auto dtype = ReadDataType(in);
				auto shape = ReadShape(in);
				(void)subgraph.AddParam(dtype, std::move(shape));
			}

			const auto nodeCount = ReadSize(in);
			if (nodeCount < paramCount)
			{
				throw std::runtime_error("LiteNN model subgraph node count is smaller than parameter count");
			}
			for (std::size_t nodeId = 0; nodeId < nodeCount; ++nodeId)
			{
				auto outputInfos = ReadOutputInfoList(in);
				auto node = ReadNodePayload(in);
				if (nodeId < paramCount)
				{
					const auto* param = std::get_if<ParamRefNode>(&node);
					if (!param || param->paramIndex != nodeId)
					{
						throw std::runtime_error("LiteNN model parameter nodes must be serialized first");
					}
					continue;
				}
				(void)subgraph.AddNode(std::move(node), std::move(outputInfos));
			}
			subgraph.SetResults(ReadNodeOutputList(in));
			return subgraph;
		}
	} // namespace Detail

	inline void SaveModel(const Graph& graph, const std::filesystem::path& path)
	{
		Validation::ValidateGraph(graph);
		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open LiteNN model file for writing");
		}

		out.write(Detail::kModelMagic.data(), static_cast<std::streamsize>(Detail::kModelMagic.size()));
		Detail::EnsureWrite(out);
		Detail::WriteScalar(out, Detail::kModelVersion);
		Detail::WriteSize(out, graph.Forward());
		Detail::WriteScalar(out, static_cast<std::uint8_t>(graph.Backward().has_value() ? 1 : 0));
		if (graph.Backward())
		{
			Detail::WriteSize(out, *graph.Backward());
		}
		Detail::WriteStringList(out, graph.InputNames());
		Detail::WriteStringList(out, graph.OutputNames());

		Detail::WriteSize(out, graph.VariableCount());
		for (const auto& variable : graph.Variables())
		{
			Detail::WriteTensor(out, variable->Data());
		}

		Detail::WriteSize(out, graph.ActivationSlotCount());
		for (std::size_t i = 0; i < graph.ActivationSlotCount(); ++i)
		{
			const auto& slot = graph.GetActivationSlot(i);
			Detail::WriteDataType(out, slot.dtype);
			Detail::WriteShape(out, slot.shape);
		}

		Detail::WriteSize(out, graph.TapeSlotCount());
		for (std::size_t i = 0; i < graph.TapeSlotCount(); ++i)
		{
			const auto& slot = graph.GetTapeSlot(i);
			Detail::WriteDataType(out, slot.dtype);
			Detail::WriteShape(out, slot.shape);
		}

		Detail::WriteSize(out, graph.SubgraphCount());
		for (SubgraphId id = 0; id < graph.SubgraphCount(); ++id)
		{
			Detail::WriteSubgraph(out, graph.GetSubgraph(id));
		}
	}

	inline Graph LoadModel(const std::filesystem::path& path)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
		{
			throw std::runtime_error("Failed to open LiteNN model file for reading");
		}

		std::array<char, Detail::kModelMagic.size()> magic{};
		in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
		Detail::EnsureRead(in);
		if (magic != Detail::kModelMagic)
		{
			throw std::runtime_error("Invalid LiteNN model magic header");
		}

		const auto version = Detail::ReadScalar<std::uint32_t>(in);
		if (version == 0 || version > Detail::kModelVersion)
		{
			throw std::runtime_error("Unsupported LiteNN model version");
		}

		const auto forward = Detail::ReadSize(in);
		const auto hasBackward = Detail::ReadScalar<std::uint8_t>(in) != 0;
		std::optional<SubgraphId> backward;
		if (hasBackward)
		{
			backward = Detail::ReadSize(in);
		}
		std::vector<std::string> inputNames;
		std::vector<std::string> outputNames;
		if (version >= 2)
		{
			inputNames = Detail::ReadStringList(in);
			outputNames = Detail::ReadStringList(in);
		}

		Graph graph;
		const auto variableCount = Detail::ReadSize(in);
		for (std::size_t i = 0; i < variableCount; ++i)
		{
			graph.AddVariable(Variable::Create(Detail::ReadTensor(in)));
		}

		const auto activationSlotCount = Detail::ReadSize(in);
		for (std::size_t i = 0; i < activationSlotCount; ++i)
		{
			graph.AddActivationSlot({ Detail::ReadDataType(in), Detail::ReadShape(in) });
		}

		const auto tapeSlotCount = Detail::ReadSize(in);
		for (std::size_t i = 0; i < tapeSlotCount; ++i)
		{
			graph.AddTapeSlot({ Detail::ReadDataType(in), Detail::ReadShape(in) });
		}

		const auto subgraphCount = Detail::ReadSize(in);
		for (std::size_t i = 0; i < subgraphCount; ++i)
		{
			graph.AddSubgraph(Detail::ReadSubgraph(in));
		}
		graph.SetForward(forward);
		if (backward)
		{
			graph.SetBackward(*backward);
		}
		graph.SetInputNames(std::move(inputNames));
		graph.SetOutputNames(std::move(outputNames));

		if (in.peek() != std::char_traits<char>::eof())
		{
			throw std::runtime_error("LiteNN model contains trailing bytes");
		}

		Validation::ValidateGraph(graph);
		return graph;
	}
} // namespace LiteNN::Serialization

#endif
