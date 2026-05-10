#pragma once

#include <LiteNN/Device.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Tensor.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN
{
	struct CompiledTensorSpec
	{
		DataType dtype{};
		std::vector<std::size_t> shape;
		std::string name;
	};

	struct CompiledModuleImage
	{
		const void* rodata{};
		std::size_t rodataSize{};
		const void* instructions{};
		std::size_t instructionSize{};
	};

	struct CompiledModuleExportedSymbols
	{
		const void* rodata{};
		const void* rodataSize{};
		const void* instructions{};
		const void* instructionSize{};
	};

	struct CompiledModuleInvocation
	{
		std::span<const Tensor<CPU>> inputs;
		std::span<Tensor<CPU>> outputs;
	};

	template <Device D>
	class CompiledModule;

	template <Device D>
	class Compiler;

	class CompiledModuleArtifact
	{
	public:
		CompiledModuleArtifact() = default;
		CompiledModuleArtifact(const CompiledModuleArtifact&) = default;
		CompiledModuleArtifact(CompiledModuleArtifact&&) noexcept = default;
		CompiledModuleArtifact& operator=(const CompiledModuleArtifact&) = default;
		CompiledModuleArtifact& operator=(CompiledModuleArtifact&&) noexcept = default;
		~CompiledModuleArtifact() = default;

		static CompiledModuleArtifact CopyFromImage(CompiledModuleImage image);
		static CompiledModuleArtifact FromExportedSymbols(CompiledModuleExportedSymbols symbols);

		/// Loads the artifact into a runnable module. The artifact remains valid after loading.
		CompiledModule<CPU> Load() const;

		CompiledModuleImage Image() const;
		std::span<const std::byte> Rodata() const;
		std::span<const std::byte> Instructions() const;
		std::span<const CompiledTensorSpec> InputSpecs() const;
		std::span<const CompiledTensorSpec> OutputSpecs() const;
		std::optional<std::size_t> FindInput(std::string_view name) const;
		std::optional<std::size_t> FindOutput(std::string_view name) const;

		void WriteObjectFile(const std::filesystem::path& path,
		                     std::string_view symbolPrefix = "litenn_module") const;

	private:
		friend class Compiler<CPU>;

		CompiledModuleArtifact(std::vector<std::byte> rodata,
		                      std::vector<std::byte> instructions,
		                      std::vector<CompiledTensorSpec> inputSpecs,
		                      std::vector<CompiledTensorSpec> outputSpecs);

		std::vector<std::byte> rodata_;
		std::vector<std::byte> instructions_;
		std::vector<CompiledTensorSpec> inputSpecs_;
		std::vector<CompiledTensorSpec> outputSpecs_;
	};

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

		/// Loads a borrowed image by copying rodata/instruction bytes into module-owned storage.
		/// The caller may release the original image memory after this returns.
		static CompiledModule Load(CompiledModuleImage image);

		/// Runs the compiled entry point and returns newly allocated output tensors.
		std::vector<Tensor<CPU>> Run(std::span<const Tensor<CPU>> inputs) const;

		/// Runs the compiled entry point into caller-provided output tensors.
		void RunInto(std::span<const Tensor<CPU>> inputs, std::span<Tensor<CPU>> outputs) const;

		/// Runs independent invocations concurrently when threadCount > 1.
		/// Concurrent Run/RunInto/RunManyInto calls are supported when each call uses
		/// independent input/output buffers.
		void RunManyInto(std::span<const CompiledModuleInvocation> invocations,
		                 std::size_t threadCount = 0) const;

		CompiledModuleImage Image() const;
		std::span<const std::byte> Rodata() const;
		std::span<const std::byte> Instructions() const;
		std::span<const CompiledTensorSpec> InputSpecs() const;
		std::span<const CompiledTensorSpec> OutputSpecs() const;
		std::optional<std::size_t> FindInput(std::string_view name) const;
		std::optional<std::size_t> FindOutput(std::string_view name) const;

		void WriteObjectFile(const std::filesystem::path& path,
		                     std::string_view symbolPrefix = "litenn_module") const;

	private:
		struct Impl;

		explicit CompiledModule(std::shared_ptr<Impl> impl);

		std::shared_ptr<Impl> impl_;
	};

	template <>
	class Compiler<CPU>
	{
	public:
		static CompiledModuleArtifact CompileArtifact(const Graph& graph);
		static CompiledModule<CPU> Compile(const Graph& graph);
	};
} // namespace LiteNN
