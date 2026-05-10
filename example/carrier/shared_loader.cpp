#include "graph_common.h"

#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <string_view>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{
	class DynamicLibrary
	{
	public:
		explicit DynamicLibrary(const std::filesystem::path& path)
		{
	#if defined(_WIN32)
			handle_ = LoadLibraryW(path.c_str());
			if (!handle_)
			{
				throw std::runtime_error(std::format("Failed to open shared library {}", path.string()));
			}
	#else
			handle_ = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
			if (!handle_)
			{
				throw std::runtime_error(std::format("Failed to open shared library {}: {}", path.string(), dlerror()));
			}
	#endif
		}

		DynamicLibrary(const DynamicLibrary&) = delete;
		DynamicLibrary& operator=(const DynamicLibrary&) = delete;

		~DynamicLibrary()
		{
	#if defined(_WIN32)
			if (handle_)
			{
				FreeLibrary(handle_);
			}
	#else
			if (handle_)
			{
				dlclose(handle_);
			}
	#endif
		}

		const void* Lookup(std::string_view name) const
		{
	#if defined(_WIN32)
			auto* address = reinterpret_cast<const void*>(GetProcAddress(handle_, std::string(name).c_str()));
			if (!address)
			{
				throw std::runtime_error(std::format("Missing exported symbol {}", name));
			}
			return address;
	#else
			dlerror();
			auto* address = dlsym(handle_, std::string(name).c_str());
			if (!address)
			{
				throw std::runtime_error(std::format("Missing exported symbol {}: {}", name, dlerror()));
			}
			return address;
	#endif
		}

	private:
	#if defined(_WIN32)
		HMODULE handle_{};
	#else
		void* handle_{};
	#endif
	};

	std::filesystem::path DefaultLibraryPath(const char* argv0)
	{
		return std::filesystem::absolute(argv0).parent_path() / LITENN_CARRIER_SHARED_FILE;
	}

	std::string SymbolName(std::string_view suffix)
	{
		return std::string(LITENN_CARRIER_SYMBOL_PREFIX) + std::string(suffix);
	}
} // namespace

int main(int argc, char** argv)
{
	using namespace LiteNN;
	using namespace LiteNN::Examples::Carrier;

	try
	{
		const auto libraryPath = argc > 1 ? std::filesystem::path(argv[1]) : DefaultLibraryPath(argv[0]);
		DynamicLibrary library(libraryPath);
		auto artifact = CompiledModuleArtifact::FromExportedSymbols({
		    .rodata = library.Lookup(SymbolName("_rodata")),
		    .rodataSize = library.Lookup(SymbolName("_rodata_size")),
		    .instructions = library.Lookup(SymbolName("_instructions")),
		    .instructionSize = library.Lookup(SymbolName("_instructions_size")),
		});
		auto module = artifact.Load();
		auto inputs = MakeSampleInputs();
		auto outputs = module.Run(inputs);
		VerifySampleOutputs(outputs);
		PrintRunSummary("Shared", module, outputs);
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "error: " << ex.what() << '\n';
		return 1;
	}
}