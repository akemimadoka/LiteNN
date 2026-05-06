#pragma once

#include <LiteNN/Device.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Tensor.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace LiteNN
{
	struct CompiledTensorSpec
	{
		DataType dtype{};
		std::vector<std::size_t> shape;
	};

	struct CompiledModuleImage
	{
		const void* rodata{};
		std::size_t rodataSize{};
		const void* instructions{};
		std::size_t instructionSize{};
	};

	template <Device D>
	class CompiledModule;

	template <>
	class CompiledModule<CPU>
	{
	public:
		CompiledModule();
		CompiledModule(const CompiledModule&);
		CompiledModule(CompiledModule&&) noexcept;
		CompiledModule& operator=(const CompiledModule&);
		CompiledModule& operator=(CompiledModule&&) noexcept;
		~CompiledModule();

		static CompiledModule Load(CompiledModuleImage image);

		std::vector<Tensor<CPU>> Run(std::span<const Tensor<CPU>> inputs) const;

		CompiledModuleImage Image() const;
		std::span<const std::byte> Rodata() const;
		std::span<const std::byte> Instructions() const;
		std::span<const CompiledTensorSpec> InputSpecs() const;
		std::span<const CompiledTensorSpec> OutputSpecs() const;

		void WriteObjectFile(const std::filesystem::path& path,
		                     std::string_view symbolPrefix = "litenn_module") const;

	private:
		struct Impl;

		explicit CompiledModule(std::shared_ptr<Impl> impl);

		std::shared_ptr<Impl> impl_;
	};

	template <Device D>
	class Compiler;

	template <>
	class Compiler<CPU>
	{
	public:
		static CompiledModule<CPU> Compile(const Graph& graph);
	};
} // namespace LiteNN
