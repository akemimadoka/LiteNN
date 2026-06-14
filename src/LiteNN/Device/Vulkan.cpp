#ifndef LITENN_MODULE_IMPL
#include "Vulkan.h"
#endif

#ifdef LITENN_ENABLE_VULKAN

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace LiteNN
{
	namespace clk = std::chrono;

	namespace
	{
		void CheckVulkan(VkResult result, std::string_view operation)
		{
			if (result != VK_SUCCESS)
			{
				throw std::runtime_error(std::format("{} failed with VkResult {}", operation, static_cast<int>(result)));
			}
		}

		std::uint64_t CheckedByteSize(DataType type, std::size_t elementCount)
		{
			const auto elementBytes = ElementByteSize(type);
			if (elementCount > std::numeric_limits<std::uint64_t>::max() / elementBytes)
			{
				throw std::runtime_error("Vulkan tensor allocation size overflows uint64_t");
			}
			return static_cast<std::uint64_t>(elementCount) * elementBytes;
		}

		VkInstance CreateLiteNNVulkanInstance()
		{
			const VkApplicationInfo appInfo{
				.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
				.pApplicationName = "LiteNN",
				.applicationVersion = VK_MAKE_VERSION(0, 1, 0),
				.pEngineName = "LiteNN",
				.engineVersion = VK_MAKE_VERSION(0, 1, 0),
				.apiVersion = VK_API_VERSION_1_1,
			};
			const VkInstanceCreateInfo createInfo{
				.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
				.pApplicationInfo = &appInfo,
			};
			VkInstance instance{};
			CheckVulkan(vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance");
			return instance;
		}

		std::uint32_t CountDevicesForInstance(VkInstance instance)
		{
			std::uint32_t count = 0;
			CheckVulkan(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices(count)");
			return count;
		}

		std::vector<VkPhysicalDevice> EnumerateDevices(VkInstance instance)
		{
			const auto count = CountDevicesForInstance(instance);
			std::vector<VkPhysicalDevice> devices(count);
			if (count != 0)
			{
				std::uint32_t writableCount = count;
				CheckVulkan(vkEnumeratePhysicalDevices(instance, &writableCount, devices.data()),
				            "vkEnumeratePhysicalDevices(list)");
			}
			return devices;
		}

		struct ComputeQueueFamilySelection
		{
			std::uint32_t index{};
			VkQueueFamilyProperties properties{};
		};

		ComputeQueueFamilySelection FindComputeQueueFamily(VkPhysicalDevice physicalDevice)
		{
			std::uint32_t count = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
			std::vector<VkQueueFamilyProperties> families(count);
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
			for (std::uint32_t i = 0; i < families.size(); ++i)
			{
				if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
				{
					return { .index = i, .properties = families[i] };
				}
			}
			throw std::runtime_error("Vulkan device has no compute queue family");
		}

		std::uint32_t FindMemoryType(VkPhysicalDevice physicalDevice, std::uint32_t typeBits,
		                             VkMemoryPropertyFlags required)
		{
			VkPhysicalDeviceMemoryProperties properties{};
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
			for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i)
			{
				const bool typeAllowed = (typeBits & (1u << i)) != 0;
				const bool hasFlags = (properties.memoryTypes[i].propertyFlags & required) == required;
				if (typeAllowed && hasFlags)
				{
					return i;
				}
			}
			throw std::runtime_error("Vulkan device has no compatible storage buffer memory type");
		}

		VkMemoryPropertyFlags BufferMemoryProperties(VulkanBufferResidency residency)
		{
			switch (residency)
			{
			case VulkanBufferResidency::HostVisibleCoherent:
				return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			case VulkanBufferResidency::DeviceLocal:
				return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			}
			throw std::runtime_error("Invalid Vulkan buffer residency");
		}

		bool VulkanApiVersionAtLeast(std::uint32_t version, std::uint32_t major, std::uint32_t minor)
		{
			const auto actualMajor = VK_VERSION_MAJOR(version);
			const auto actualMinor = VK_VERSION_MINOR(version);
			return actualMajor > major || (actualMajor == major && actualMinor >= minor);
		}

		std::vector<VkExtensionProperties> EnumerateDeviceExtensions(VkPhysicalDevice physicalDevice)
		{
			std::uint32_t count = 0;
			CheckVulkan(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr),
			            "vkEnumerateDeviceExtensionProperties(count)");
			std::vector<VkExtensionProperties> extensions(count);
			if (count != 0)
			{
				CheckVulkan(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data()),
				            "vkEnumerateDeviceExtensionProperties(list)");
			}
			return extensions;
		}

		bool HasDeviceExtension(std::span<const VkExtensionProperties> extensions, std::string_view name)
		{
			return std::ranges::any_of(extensions, [&](const VkExtensionProperties& extension) {
				return std::string_view{ extension.extensionName } == name;
			});
		}

		void AppendUniqueExtension(std::vector<const char*>& extensions, const char* name)
		{
			if (std::ranges::find_if(extensions, [&](const char* existing) {
				    return std::string_view{ existing } == name;
			    }) == extensions.end())
			{
				extensions.push_back(name);
			}
		}
	} // namespace

	class VulkanContext
	{
	public:
		explicit VulkanContext(std::uint32_t requestedDeviceIndex) : deviceIndex(requestedDeviceIndex)
		{
			instance = CreateLiteNNVulkanInstance();
			auto devices = EnumerateDevices(instance);
			if (deviceIndex >= devices.size())
			{
				throw std::runtime_error(std::format("Vulkan device index {} is out of range; {} device(s) available",
				                                     deviceIndex, devices.size()));
			}
			physicalDevice = devices[deviceIndex];
			VkPhysicalDeviceProperties2 properties2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
			void** propertiesNext = &properties2.pNext;
#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES)
			VkPhysicalDeviceSubgroupProperties subgroupProperties{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
			};
			*propertiesNext = &subgroupProperties;
			propertiesNext = &subgroupProperties.pNext;
#endif
#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES)
			VkPhysicalDeviceDescriptorIndexingProperties descriptorIndexingProperties{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
			};
			*propertiesNext = &descriptorIndexingProperties;
			propertiesNext = &descriptorIndexingProperties.pNext;
#endif
			(void)propertiesNext;
			vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
			properties = properties2.properties;
			capabilities.apiVersionMajor = VK_VERSION_MAJOR(properties.apiVersion);
			capabilities.apiVersionMinor = VK_VERSION_MINOR(properties.apiVersion);
			capabilities.apiVersionPatch = VK_VERSION_PATCH(properties.apiVersion);
			capabilities.maxComputeWorkGroupInvocations = properties.limits.maxComputeWorkGroupInvocations;
			capabilities.maxComputeWorkGroupSize = {
				properties.limits.maxComputeWorkGroupSize[0],
				properties.limits.maxComputeWorkGroupSize[1],
				properties.limits.maxComputeWorkGroupSize[2],
			};
			capabilities.maxComputeWorkGroupCount = {
				properties.limits.maxComputeWorkGroupCount[0],
				properties.limits.maxComputeWorkGroupCount[1],
				properties.limits.maxComputeWorkGroupCount[2],
			};
			capabilities.maxStorageBufferRange = properties.limits.maxStorageBufferRange;
			capabilities.minStorageBufferOffsetAlignment = properties.limits.minStorageBufferOffsetAlignment;
			capabilities.maxPerStageResources = properties.limits.maxPerStageResources;
			capabilities.maxComputeSharedMemorySize = properties.limits.maxComputeSharedMemorySize;
			capabilities.maxPushConstantsSize = properties.limits.maxPushConstantsSize;
			capabilities.maxPerStageDescriptorStorageBuffers =
			    properties.limits.maxPerStageDescriptorStorageBuffers;
			capabilities.maxDescriptorSetStorageBuffers = properties.limits.maxDescriptorSetStorageBuffers;
			capabilities.maxBoundDescriptorSets = properties.limits.maxBoundDescriptorSets;
			capabilities.deviceName = properties.deviceName;
#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES)
			capabilities.subgroupSize = subgroupProperties.subgroupSize;
			capabilities.subgroupComputeAvailable =
			    (subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
			capabilities.subgroupBasicAvailable =
			    (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
			capabilities.subgroupArithmeticAvailable =
			    (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
			capabilities.subgroupVoteAvailable =
			    (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0;
			capabilities.subgroupBallotAvailable =
			    (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0;
			capabilities.subgroupShuffleAvailable =
			    (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0;
			capabilities.subgroupShuffleRelativeAvailable =
			    (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0;
			capabilities.subgroupClusteredAvailable =
			    (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0;
			capabilities.subgroupQuadAvailable =
			    (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) != 0;
#endif
#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES)
			capabilities.maxUpdateAfterBindDescriptorsInAllPools =
			    descriptorIndexingProperties.maxUpdateAfterBindDescriptorsInAllPools;
			capabilities.maxPerStageDescriptorUpdateAfterBindStorageBuffers =
			    descriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindStorageBuffers;
			capabilities.maxDescriptorSetUpdateAfterBindStorageBuffers =
			    descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageBuffers;
			capabilities.maxDescriptorSetVariableDescriptorCount =
			    descriptorIndexingProperties.maxDescriptorSetVariableDescriptorCount;
#endif

			auto deviceExtensions = EnumerateDeviceExtensions(physicalDevice);
			const auto apiAtLeast11 = VulkanApiVersionAtLeast(properties.apiVersion, 1, 1);
			const auto apiAtLeast12 = VulkanApiVersionAtLeast(properties.apiVersion, 1, 2);

			VkPhysicalDeviceFeatures2 availableFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
			void** availableNext = &availableFeatures.pNext;
#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES)
			VkPhysicalDevice16BitStorageFeatures availableStorage16{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
			};
			*availableNext = &availableStorage16;
			availableNext = &availableStorage16.pNext;
#endif
#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES)
			VkPhysicalDevice8BitStorageFeatures availableStorage8{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
			};
			*availableNext = &availableStorage8;
			availableNext = &availableStorage8.pNext;
#endif
#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
			VkPhysicalDeviceShaderFloat16Int8Features availableShaderFloat16Int8{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
			};
			*availableNext = &availableShaderFloat16Int8;
			availableNext = &availableShaderFloat16Int8.pNext;
#endif
#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
			VkPhysicalDeviceDescriptorIndexingFeatures availableDescriptorIndexing{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
			};
			*availableNext = &availableDescriptorIndexing;
			availableNext = &availableDescriptorIndexing.pNext;
#endif
			(void)availableNext;
			vkGetPhysicalDeviceFeatures2(physicalDevice, &availableFeatures);

			VkPhysicalDeviceFeatures2 enabledFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
			void** enabledNext = &enabledFeatures.pNext;
			std::vector<const char*> enabledDeviceExtensions;

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES)
			capabilities.storageBuffer16BitAccessAvailable = availableStorage16.storageBuffer16BitAccess == VK_TRUE;
			VkPhysicalDevice16BitStorageFeatures enabledStorage16{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
			};
			const bool canEnableStorage16 =
			    capabilities.storageBuffer16BitAccessAvailable &&
			    (apiAtLeast11
#if defined(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)
			     || HasDeviceExtension(deviceExtensions, VK_KHR_16BIT_STORAGE_EXTENSION_NAME)
#endif
			    );
			if (canEnableStorage16)
			{
#if defined(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)
				if (!apiAtLeast11)
				{
					AppendUniqueExtension(enabledDeviceExtensions, VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
				}
#endif
				enabledStorage16.storageBuffer16BitAccess = VK_TRUE;
				capabilities.storageBuffer16BitAccessEnabled = true;
				*enabledNext = &enabledStorage16;
				enabledNext = &enabledStorage16.pNext;
			}
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES)
			capabilities.storageBuffer8BitAccessAvailable = availableStorage8.storageBuffer8BitAccess == VK_TRUE;
			VkPhysicalDevice8BitStorageFeatures enabledStorage8{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
			};
			const bool canEnableStorage8 =
			    capabilities.storageBuffer8BitAccessAvailable &&
			    (apiAtLeast12
#if defined(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)
			     || HasDeviceExtension(deviceExtensions, VK_KHR_8BIT_STORAGE_EXTENSION_NAME)
#endif
			    );
			if (canEnableStorage8)
			{
#if defined(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)
				if (!apiAtLeast12)
				{
					AppendUniqueExtension(enabledDeviceExtensions, VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
				}
#endif
				enabledStorage8.storageBuffer8BitAccess = VK_TRUE;
				capabilities.storageBuffer8BitAccessEnabled = true;
				*enabledNext = &enabledStorage8;
				enabledNext = &enabledStorage8.pNext;
			}
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
			capabilities.shaderFloat16Available = availableShaderFloat16Int8.shaderFloat16 == VK_TRUE;
			capabilities.shaderInt8Available = availableShaderFloat16Int8.shaderInt8 == VK_TRUE;
			VkPhysicalDeviceShaderFloat16Int8Features enabledShaderFloat16Int8{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
			};
			const bool canEnableShaderFloat16Int8 =
			    (capabilities.shaderFloat16Available || capabilities.shaderInt8Available) &&
			    (apiAtLeast12
#if defined(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)
			     || HasDeviceExtension(deviceExtensions, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)
#endif
			    );
			if (canEnableShaderFloat16Int8)
			{
#if defined(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)
				if (!apiAtLeast12)
				{
					AppendUniqueExtension(enabledDeviceExtensions, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
				}
#endif
				enabledShaderFloat16Int8.shaderFloat16 =
				    capabilities.shaderFloat16Available ? VK_TRUE : VK_FALSE;
				enabledShaderFloat16Int8.shaderInt8 = capabilities.shaderInt8Available ? VK_TRUE : VK_FALSE;
				capabilities.shaderFloat16Enabled = enabledShaderFloat16Int8.shaderFloat16 == VK_TRUE;
				capabilities.shaderInt8Enabled = enabledShaderFloat16Int8.shaderInt8 == VK_TRUE;
				*enabledNext = &enabledShaderFloat16Int8;
				enabledNext = &enabledShaderFloat16Int8.pNext;
			}
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
			capabilities.shaderStorageBufferArrayNonUniformIndexingAvailable =
			    availableDescriptorIndexing.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE;
			capabilities.descriptorBindingStorageBufferUpdateAfterBindAvailable =
			    availableDescriptorIndexing.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE;
			capabilities.descriptorBindingPartiallyBoundAvailable =
			    availableDescriptorIndexing.descriptorBindingPartiallyBound == VK_TRUE;
			capabilities.descriptorBindingVariableDescriptorCountAvailable =
			    availableDescriptorIndexing.descriptorBindingVariableDescriptorCount == VK_TRUE;
			capabilities.runtimeDescriptorArrayAvailable =
			    availableDescriptorIndexing.runtimeDescriptorArray == VK_TRUE;
			VkPhysicalDeviceDescriptorIndexingFeatures enabledDescriptorIndexing{
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
			};
			const bool canEnableDescriptorIndexing =
			    (capabilities.shaderStorageBufferArrayNonUniformIndexingAvailable ||
			     capabilities.descriptorBindingStorageBufferUpdateAfterBindAvailable ||
			     capabilities.descriptorBindingPartiallyBoundAvailable ||
			     capabilities.descriptorBindingVariableDescriptorCountAvailable ||
			     capabilities.runtimeDescriptorArrayAvailable) &&
			    (apiAtLeast12
#if defined(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
			     || HasDeviceExtension(deviceExtensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
#endif
			    );
			if (canEnableDescriptorIndexing)
			{
#if defined(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
				if (!apiAtLeast12)
				{
					AppendUniqueExtension(enabledDeviceExtensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
				}
#endif
				enabledDescriptorIndexing.shaderStorageBufferArrayNonUniformIndexing =
				    capabilities.shaderStorageBufferArrayNonUniformIndexingAvailable ? VK_TRUE : VK_FALSE;
				enabledDescriptorIndexing.descriptorBindingStorageBufferUpdateAfterBind =
				    capabilities.descriptorBindingStorageBufferUpdateAfterBindAvailable ? VK_TRUE : VK_FALSE;
				enabledDescriptorIndexing.descriptorBindingPartiallyBound =
				    capabilities.descriptorBindingPartiallyBoundAvailable ? VK_TRUE : VK_FALSE;
				enabledDescriptorIndexing.descriptorBindingVariableDescriptorCount =
				    capabilities.descriptorBindingVariableDescriptorCountAvailable ? VK_TRUE : VK_FALSE;
				enabledDescriptorIndexing.runtimeDescriptorArray =
				    capabilities.runtimeDescriptorArrayAvailable ? VK_TRUE : VK_FALSE;
				capabilities.shaderStorageBufferArrayNonUniformIndexingEnabled =
				    enabledDescriptorIndexing.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE;
				capabilities.descriptorBindingStorageBufferUpdateAfterBindEnabled =
				    enabledDescriptorIndexing.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE;
				capabilities.descriptorBindingPartiallyBoundEnabled =
				    enabledDescriptorIndexing.descriptorBindingPartiallyBound == VK_TRUE;
				capabilities.descriptorBindingVariableDescriptorCountEnabled =
				    enabledDescriptorIndexing.descriptorBindingVariableDescriptorCount == VK_TRUE;
				capabilities.runtimeDescriptorArrayEnabled =
				    enabledDescriptorIndexing.runtimeDescriptorArray == VK_TRUE;
				*enabledNext = &enabledDescriptorIndexing;
				enabledNext = &enabledDescriptorIndexing.pNext;
			}
#endif
			(void)enabledNext;

			const auto computeQueue = FindComputeQueueFamily(physicalDevice);
			queueFamilyIndex = computeQueue.index;
			capabilities.timestampPeriodNanoseconds = properties.limits.timestampPeriod;
			capabilities.computeQueueTimestampValidBits = computeQueue.properties.timestampValidBits;
			capabilities.computeQueueTimestampsAvailable =
			    computeQueue.properties.timestampValidBits != 0 && properties.limits.timestampPeriod > 0.0f;

			const float queuePriority = 1.0f;
			const VkDeviceQueueCreateInfo queueInfo{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = queueFamilyIndex,
				.queueCount = 1,
				.pQueuePriorities = &queuePriority,
			};
			const void* enabledFeatureChain = enabledFeatures.pNext == nullptr ? nullptr : &enabledFeatures;
			const VkDeviceCreateInfo deviceInfo{
				.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
				.pNext = enabledFeatureChain,
				.queueCreateInfoCount = 1,
				.pQueueCreateInfos = &queueInfo,
				.enabledExtensionCount = static_cast<std::uint32_t>(enabledDeviceExtensions.size()),
				.ppEnabledExtensionNames =
				    enabledDeviceExtensions.empty() ? nullptr : enabledDeviceExtensions.data(),
			};
			CheckVulkan(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device), "vkCreateDevice");
			vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

			const VkCommandPoolCreateInfo poolInfo{
				.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
				.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
				.queueFamilyIndex = queueFamilyIndex,
			};
			CheckVulkan(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "vkCreateCommandPool");

			const VkPipelineCacheCreateInfo pipelineCacheInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
			};
			CheckVulkan(vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &pipelineCache),
			            "vkCreatePipelineCache");
		}

		VulkanContext(const VulkanContext&) = delete;
		VulkanContext& operator=(const VulkanContext&) = delete;

		~VulkanContext()
		{
			if (device != VK_NULL_HANDLE)
			{
				(void)vkDeviceWaitIdle(device);
				if (commandPool != VK_NULL_HANDLE)
				{
					vkDestroyCommandPool(device, commandPool, nullptr);
				}
				if (pipelineCache != VK_NULL_HANDLE)
				{
					vkDestroyPipelineCache(device, pipelineCache, nullptr);
				}
				vkDestroyDevice(device, nullptr);
			}
			if (instance != VK_NULL_HANDLE)
			{
				vkDestroyInstance(instance, nullptr);
			}
		}

		std::uint32_t deviceIndex{};
		VkInstance instance{};
		VkPhysicalDevice physicalDevice{};
		VkDevice device{};
		VkQueue queue{};
		std::uint32_t queueFamilyIndex{};
		VkCommandPool commandPool{};
		VkPipelineCache pipelineCache{};
		VkPhysicalDeviceProperties properties{};
		VulkanDeviceCapabilities capabilities{};
		std::mutex queueMutex;
	};

	namespace
	{
		std::shared_ptr<VulkanContext> GetContext(const Vulkan& device)
		{
			if (!device.context)
			{
				static std::mutex cacheMutex;
				static std::unordered_map<std::uint32_t, std::weak_ptr<VulkanContext>> cache;
				std::lock_guard lock(cacheMutex);
				if (auto existing = cache[device.deviceIndex].lock())
				{
					device.context = std::move(existing);
				}
				else
				{
					device.context = std::make_shared<VulkanContext>(device.deviceIndex);
					cache[device.deviceIndex] = device.context;
				}
			}
			return device.context;
		}

		struct VulkanBuffer
		{
			std::shared_ptr<VulkanContext> context;
			VkBuffer buffer{};
			VkDeviceMemory memory{};
			VkDeviceSize byteSize{};
			VulkanBufferResidency residency{ VulkanBufferResidency::HostVisibleCoherent };

			VulkanBuffer(std::shared_ptr<VulkanContext> ctx, VkDeviceSize size, VulkanBufferResidency residency)
			    : context(std::move(ctx)), byteSize(size), residency(residency)
			{
				const VkBufferCreateInfo bufferInfo{
					.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
					.size = std::max<VkDeviceSize>(1, byteSize),
					.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
					          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				};
				CheckVulkan(vkCreateBuffer(context->device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");

				VkMemoryRequirements requirements{};
				vkGetBufferMemoryRequirements(context->device, buffer, &requirements);
				const auto memoryType = FindMemoryType(context->physicalDevice, requirements.memoryTypeBits,
				                                       BufferMemoryProperties(residency));
				const VkMemoryAllocateInfo allocateInfo{
					.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
					.allocationSize = requirements.size,
					.memoryTypeIndex = memoryType,
				};
				CheckVulkan(vkAllocateMemory(context->device, &allocateInfo, nullptr, &memory), "vkAllocateMemory");
				CheckVulkan(vkBindBufferMemory(context->device, buffer, memory, 0), "vkBindBufferMemory");
			}

			VulkanBuffer(const VulkanBuffer&) = delete;
			VulkanBuffer& operator=(const VulkanBuffer&) = delete;

			~VulkanBuffer()
			{
				if (context)
				{
					if (buffer != VK_NULL_HANDLE)
					{
						vkDestroyBuffer(context->device, buffer, nullptr);
					}
					if (memory != VK_NULL_HANDLE)
					{
						vkFreeMemory(context->device, memory, nullptr);
					}
				}
			}
		};

		VulkanBuffer& AsBuffer(void* ptr)
		{
			if (!ptr)
			{
				throw std::runtime_error("Vulkan buffer pointer is null");
			}
			return *static_cast<VulkanBuffer*>(ptr);
		}

		const VulkanBuffer& AsBuffer(const void* ptr)
		{
			if (!ptr)
			{
				throw std::runtime_error("Vulkan buffer pointer is null");
			}
			return *static_cast<const VulkanBuffer*>(ptr);
		}

		void* MapBuffer(VulkanBuffer& buffer)
		{
			if (buffer.residency != VulkanBufferResidency::HostVisibleCoherent)
			{
				throw std::runtime_error("Vulkan buffer is device-local and cannot be host-mapped yet");
			}
			void* mapped = nullptr;
			CheckVulkan(vkMapMemory(buffer.context->device, buffer.memory, 0, buffer.byteSize, 0, &mapped),
			            "vkMapMemory");
			return mapped;
		}

		const void* MapBufferForRead(const VulkanBuffer& buffer)
		{
			if (buffer.residency != VulkanBufferResidency::HostVisibleCoherent)
			{
				throw std::runtime_error("Vulkan buffer is device-local and cannot be host-mapped yet");
			}
			void* mapped = nullptr;
			CheckVulkan(vkMapMemory(buffer.context->device, buffer.memory, 0, buffer.byteSize, 0, &mapped),
			            "vkMapMemory");
			return mapped;
		}

		void UnmapBuffer(const VulkanBuffer& buffer)
		{
			vkUnmapMemory(buffer.context->device, buffer.memory);
		}

		std::string UnsupportedVulkanEagerOp(std::string_view op)
		{
			return std::format(
			    "Vulkan eager {} is not implemented; use Compiler<Vulkan> for supported native AOT kernels or enable an explicit host fallback path",
			    op);
		}
	} // namespace

	std::uint32_t VulkanDeviceCount() noexcept
	{
		try
		{
			auto instance = CreateLiteNNVulkanInstance();
			const auto count = CountDevicesForInstance(instance);
			vkDestroyInstance(instance, nullptr);
			return count;
		}
		catch (...)
		{
			return 0;
		}
	}

	bool IsVulkanDeviceAvailable(std::uint32_t deviceIndex) noexcept
	{
		return deviceIndex < VulkanDeviceCount();
	}

	VulkanDeviceCapabilities QueryVulkanDeviceCapabilities(const Vulkan& device)
	{
		const auto context = GetContext(device);
		return context->capabilities;
	}

	std::string_view DeviceTraits<Vulkan>::Name()
	{
		return "Vulkan";
	}

	std::string_view DeviceTraits<Vulkan>::Info(const Vulkan& device)
	{
		try
		{
			const auto context = GetContext(device);
			device.infoCache =
			    std::format("Vulkan device {}: {}", device.deviceIndex, context->properties.deviceName);
		}
		catch (const std::exception& ex)
		{
			device.infoCache = std::format("Vulkan unavailable: {}", ex.what());
		}
		return device.infoCache;
	}

	void* DeviceTraits<Vulkan>::Allocate(Vulkan& device, DataType type, std::size_t size)
	{
		return new VulkanBuffer(GetContext(device), CheckedByteSize(type, size), device.bufferResidency);
	}

	void DeviceTraits<Vulkan>::Deallocate(Vulkan& device, void* ptr, DataType type, std::size_t size)
	{
		(void)device;
		(void)type;
		(void)size;
		delete static_cast<VulkanBuffer*>(ptr);
	}

	void DeviceTraits<Vulkan>::ZeroFill(Vulkan& device, void* ptr, DataType type, std::size_t size)
	{
		(void)device;
		auto& buffer = AsBuffer(ptr);
		const auto bytes = CheckedByteSize(type, size);
		if (bytes > buffer.byteSize)
		{
			throw std::runtime_error("Vulkan ZeroFill byte size exceeds buffer allocation");
		}
		auto* mapped = static_cast<std::byte*>(MapBuffer(buffer));
		std::fill(mapped, mapped + static_cast<std::ptrdiff_t>(bytes), std::byte{ 0 });
		UnmapBuffer(buffer);
	}

	void DeviceTraits<Vulkan>::CopyToCPU(Vulkan& device, DataType srcType, const void* src, std::size_t size,
	                                     DataType dstType, void* dst)
	{
		(void)device;
		const auto& buffer = AsBuffer(src);
		const auto srcBytes = CheckedByteSize(srcType, size);
		if (srcBytes > buffer.byteSize)
		{
			throw std::runtime_error("Vulkan CopyToCPU byte size exceeds buffer allocation");
		}

		const auto* mapped = static_cast<const std::byte*>(MapBufferForRead(buffer));
		if (srcType == dstType)
		{
			std::memcpy(dst, mapped, static_cast<std::size_t>(srcBytes));
		}
		else
		{
			CPU cpu;
			DeviceTraits<CPU>::ConvertTo(cpu, srcType, mapped, size, dstType, dst);
		}
		UnmapBuffer(buffer);
	}

	void DeviceTraits<Vulkan>::CopyFromCPU(Vulkan& device, DataType dstType, void* dst, DataType srcType,
	                                       const void* src, std::size_t size)
	{
		(void)device;
		auto& buffer = AsBuffer(dst);
		const auto dstBytes = CheckedByteSize(dstType, size);
		if (dstBytes > buffer.byteSize)
		{
			throw std::runtime_error("Vulkan CopyFromCPU byte size exceeds buffer allocation");
		}

		auto* mapped = static_cast<std::byte*>(MapBuffer(buffer));
		if (srcType == dstType)
		{
			std::memcpy(mapped, src, static_cast<std::size_t>(dstBytes));
		}
		else
		{
			CPU cpu;
			DeviceTraits<CPU>::ConvertTo(cpu, srcType, src, size, dstType, mapped);
		}
		UnmapBuffer(buffer);
	}

	void DeviceTraits<Vulkan>::ConvertTo(Vulkan& device, DataType srcType, const void* src, std::size_t size,
	                                     DataType dstType, void* dst)
	{
		(void)device;
		const auto& srcBuffer = AsBuffer(src);
		auto& dstBuffer = AsBuffer(dst);
		const auto srcBytes = CheckedByteSize(srcType, size);
		const auto dstBytes = CheckedByteSize(dstType, size);
		if (srcBytes > srcBuffer.byteSize || dstBytes > dstBuffer.byteSize)
		{
			throw std::runtime_error("Vulkan ConvertTo byte size exceeds buffer allocation");
		}

		const auto* srcMapped = static_cast<const std::byte*>(MapBufferForRead(srcBuffer));
		auto* dstMapped = static_cast<std::byte*>(MapBuffer(dstBuffer));
		if (srcType == dstType)
		{
			std::memcpy(dstMapped, srcMapped, static_cast<std::size_t>(srcBytes));
		}
		else
		{
			CPU cpu;
			DeviceTraits<CPU>::ConvertTo(cpu, srcType, srcMapped, size, dstType, dstMapped);
		}
		UnmapBuffer(dstBuffer);
		UnmapBuffer(srcBuffer);
	}

	void DeviceTraits<Vulkan>::DoUnaryOp(Vulkan& device, UnaryOp unaryOp, void* dst, DataType type, ShapeView shape,
	                                     const void* src)
	{
		(void)device;
		(void)unaryOp;
		(void)dst;
		(void)type;
		(void)shape;
		(void)src;
		throw std::runtime_error(UnsupportedVulkanEagerOp("unary op"));
	}

	void DeviceTraits<Vulkan>::DoBinaryOp(Vulkan& device, BinaryOp binaryOp, void* dst, DataType type1,
	                                      ShapeView shape1, const void* src1, DataType type2, ShapeView shape2,
	                                      const void* src2)
	{
		(void)device;
		(void)binaryOp;
		(void)dst;
		(void)type1;
		(void)shape1;
		(void)src1;
		(void)type2;
		(void)shape2;
		(void)src2;
		throw std::runtime_error(UnsupportedVulkanEagerOp("binary op"));
	}

	void DeviceTraits<Vulkan>::DoReduceOp(Vulkan& device, ReduceOp reduceOp, void* dst, DataType type,
	                                      ShapeView shape, const void* src, std::size_t axis)
	{
		(void)device;
		(void)reduceOp;
		(void)dst;
		(void)type;
		(void)shape;
		(void)src;
		(void)axis;
		throw std::runtime_error(UnsupportedVulkanEagerOp("reduce op"));
	}

	void DeviceTraits<Vulkan>::DoConcatOp(Vulkan& device, void* dst, DataType type, const void* const* srcPtrs,
	                                      const ShapeView* srcShapes, std::size_t inputCount, std::size_t axis)
	{
		(void)device;
		(void)dst;
		(void)type;
		(void)srcPtrs;
		(void)srcShapes;
		(void)inputCount;
		(void)axis;
		throw std::runtime_error(UnsupportedVulkanEagerOp("concat op"));
	}

	void DeviceTraits<Vulkan>::DoSliceOp(Vulkan& device, void* dst, DataType type, ShapeView srcShape,
	                                     const void* src, std::size_t axis, std::size_t start, std::size_t length)
	{
		(void)device;
		(void)dst;
		(void)type;
		(void)srcShape;
		(void)src;
		(void)axis;
		(void)start;
		(void)length;
		throw std::runtime_error(UnsupportedVulkanEagerOp("slice op"));
	}

	void DeviceTraits<Vulkan>::DoGetRowsOp(Vulkan& device, void* dst, DataType dataType, ShapeView dataShape,
	                                      const void* data, DataType indexType, ShapeView indexShape,
	                                      const void* indices)
	{
		(void)device;
		(void)dst;
		(void)dataType;
		(void)dataShape;
		(void)data;
		(void)indexType;
		(void)indexShape;
		(void)indices;
		throw std::runtime_error(UnsupportedVulkanEagerOp("get_rows op"));
	}

	void DeviceTraits<Vulkan>::DoPermuteOp(Vulkan& device, void* dst, DataType type, ShapeView srcShape,
	                                       const void* src, ShapeView permutation)
	{
		(void)device;
		(void)dst;
		(void)type;
		(void)srcShape;
		(void)src;
		(void)permutation;
		throw std::runtime_error(UnsupportedVulkanEagerOp("permute op"));
	}

	struct VulkanComputeModule::Impl
	{
		Vulkan deviceHandle;
		std::shared_ptr<VulkanContext> context;
		VkShaderModule shaderModule{};
		VkDescriptorSetLayout descriptorSetLayout{};
		VkDescriptorPool descriptorPool{};
		VkPipelineLayout pipelineLayout{};
		VkPipeline pipeline{};
		std::uint32_t descriptorCount{};
		double creationWallTimeMs{};
	};

	struct VulkanQueryPoolGuard
	{
		VkDevice device{};
		VkQueryPool queryPool{};

		VulkanQueryPoolGuard() = default;
		explicit VulkanQueryPoolGuard(VkDevice deviceHandle) : device(deviceHandle)
		{
		}
		VulkanQueryPoolGuard(const VulkanQueryPoolGuard&) = delete;
		VulkanQueryPoolGuard& operator=(const VulkanQueryPoolGuard&) = delete;

		~VulkanQueryPoolGuard()
		{
			if (device != VK_NULL_HANDLE && queryPool != VK_NULL_HANDLE)
			{
				vkDestroyQueryPool(device, queryPool, nullptr);
			}
		}
	};

	std::uint64_t VulkanTimestampDelta(std::uint64_t begin, std::uint64_t end, std::uint32_t validBits)
	{
		if (validBits >= 64)
		{
			return end - begin;
		}
		const auto mask = (1ull << validBits) - 1;
		return (end - begin) & mask;
	}

	VulkanComputeModule::VulkanComputeModule() = default;

	VulkanComputeModule::VulkanComputeModule(Vulkan device, std::span<const std::uint32_t> spirv,
	                                         std::string_view entryPoint, std::uint32_t descriptorCount)
	    : impl_(std::make_unique<Impl>())
	{
		const auto creationBegin = clk::steady_clock::now();
		if (spirv.empty())
		{
			throw std::runtime_error("Vulkan compute module SPIR-V must not be empty");
		}
		if (entryPoint.empty())
		{
			throw std::runtime_error("Vulkan compute module entry point must not be empty");
		}
		if (descriptorCount == 0)
		{
			throw std::runtime_error("Vulkan compute module requires at least one descriptor binding");
		}

		impl_->deviceHandle = std::move(device);
		impl_->context = GetContext(impl_->deviceHandle);
		impl_->descriptorCount = descriptorCount;
		const auto& capabilities = impl_->context->capabilities;
		if (capabilities.maxBoundDescriptorSets < 1)
		{
			throw std::runtime_error(std::format(
			    "Vulkan compute module requires one descriptor set, but device '{}' reports maxBoundDescriptorSets={}",
			    capabilities.deviceName, capabilities.maxBoundDescriptorSets));
		}
		if (descriptorCount > capabilities.maxPerStageDescriptorStorageBuffers ||
		    descriptorCount > capabilities.maxDescriptorSetStorageBuffers)
		{
			throw std::runtime_error(std::format(
			    "Vulkan compute module requires {} storage-buffer descriptor(s), but device '{}' reports "
			    "maxPerStageDescriptorStorageBuffers={} and maxDescriptorSetStorageBuffers={}",
			    descriptorCount, capabilities.deviceName, capabilities.maxPerStageDescriptorStorageBuffers,
			    capabilities.maxDescriptorSetStorageBuffers));
		}

		const VkShaderModuleCreateInfo shaderInfo{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = spirv.size_bytes(),
			.pCode = spirv.data(),
		};
		CheckVulkan(vkCreateShaderModule(impl_->context->device, &shaderInfo, nullptr, &impl_->shaderModule),
		            "vkCreateShaderModule");

		std::vector<VkDescriptorSetLayoutBinding> bindings;
		bindings.reserve(descriptorCount);
		for (std::uint32_t i = 0; i < descriptorCount; ++i)
		{
			bindings.push_back({
			    .binding = i,
			    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			    .descriptorCount = 1,
			    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			});
		}
		const VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = static_cast<std::uint32_t>(bindings.size()),
			.pBindings = bindings.data(),
		};
		CheckVulkan(vkCreateDescriptorSetLayout(impl_->context->device, &descriptorLayoutInfo, nullptr,
		                                        &impl_->descriptorSetLayout),
		            "vkCreateDescriptorSetLayout");
		const VkDescriptorPoolSize poolSize{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = descriptorCount,
		};
		const VkDescriptorPoolCreateInfo poolInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = 1,
			.poolSizeCount = 1,
			.pPoolSizes = &poolSize,
		};
		CheckVulkan(vkCreateDescriptorPool(impl_->context->device, &poolInfo, nullptr, &impl_->descriptorPool),
		            "vkCreateDescriptorPool");

		const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &impl_->descriptorSetLayout,
		};
		CheckVulkan(vkCreatePipelineLayout(impl_->context->device, &pipelineLayoutInfo, nullptr,
		                                   &impl_->pipelineLayout),
		            "vkCreatePipelineLayout");

		const std::string entryName(entryPoint);
		const VkPipelineShaderStageCreateInfo stageInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = impl_->shaderModule,
			.pName = entryName.c_str(),
		};
		const VkComputePipelineCreateInfo pipelineInfo{
			.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.stage = stageInfo,
			.layout = impl_->pipelineLayout,
		};
		CheckVulkan(vkCreateComputePipelines(impl_->context->device, impl_->context->pipelineCache, 1, &pipelineInfo, nullptr,
		                                     &impl_->pipeline),
		            "vkCreateComputePipelines");
		const auto creationEnd = clk::steady_clock::now();
		impl_->creationWallTimeMs = clk::duration<double, std::milli>(creationEnd - creationBegin).count();
	}

	VulkanComputeModule::VulkanComputeModule(VulkanComputeModule&&) noexcept = default;
	VulkanComputeModule& VulkanComputeModule::operator=(VulkanComputeModule&&) noexcept = default;

	VulkanComputeModule::~VulkanComputeModule()
	{
		if (!impl_ || !impl_->context)
		{
			return;
		}
		if (impl_->pipeline != VK_NULL_HANDLE)
		{
			vkDestroyPipeline(impl_->context->device, impl_->pipeline, nullptr);
		}
		if (impl_->pipelineLayout != VK_NULL_HANDLE)
		{
			vkDestroyPipelineLayout(impl_->context->device, impl_->pipelineLayout, nullptr);
		}
		if (impl_->descriptorPool != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorPool(impl_->context->device, impl_->descriptorPool, nullptr);
		}
		if (impl_->descriptorSetLayout != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorSetLayout(impl_->context->device, impl_->descriptorSetLayout, nullptr);
		}
		if (impl_->shaderModule != VK_NULL_HANDLE)
		{
			vkDestroyShaderModule(impl_->context->device, impl_->shaderModule, nullptr);
		}
	}

	bool VulkanComputeModule::Empty() const noexcept
	{
		return !impl_ || impl_->pipeline == VK_NULL_HANDLE;
	}

	double VulkanComputeModule::CreationWallTimeMs() const noexcept
	{
		return impl_ ? impl_->creationWallTimeMs : 0.0;
	}

	void VulkanComputeModule::Dispatch(std::span<const void*> descriptorBuffers, VulkanDispatchDim groups,
	                                   VulkanExecutionOptions options) const
	{
		if (options.timing != nullptr)
		{
			*options.timing = {};
		}
		if (Empty())
		{
			throw std::runtime_error("Vulkan compute module is empty");
		}
		if (!options.synchronize)
		{
			throw std::runtime_error("Vulkan P0 compute module requires synchronous dispatch");
		}
		if (groups.x == 0 || groups.y == 0 || groups.z == 0)
		{
			throw std::runtime_error("Vulkan dispatch dimensions must be non-zero");
		}
		if (descriptorBuffers.size() != impl_->descriptorCount)
		{
			throw std::runtime_error(std::format("Vulkan descriptor count mismatch: expected {}, got {}",
			                                     impl_->descriptorCount, descriptorBuffers.size()));
		}
		const auto& capabilities = impl_->context->capabilities;
		if (groups.x > capabilities.maxComputeWorkGroupCount[0] ||
		    groups.y > capabilities.maxComputeWorkGroupCount[1] ||
		    groups.z > capabilities.maxComputeWorkGroupCount[2])
		{
			throw std::runtime_error(std::format(
			    "Vulkan dispatch groups {}x{}x{} exceed device '{}' maxComputeWorkGroupCount {}x{}x{}",
			    groups.x, groups.y, groups.z, capabilities.deviceName, capabilities.maxComputeWorkGroupCount[0],
			    capabilities.maxComputeWorkGroupCount[1], capabilities.maxComputeWorkGroupCount[2]));
		}
		const bool recordGpuTimestamp =
		    options.timing != nullptr && capabilities.computeQueueTimestampsAvailable;

		std::lock_guard lock(impl_->context->queueMutex);

		CheckVulkan(vkResetDescriptorPool(impl_->context->device, impl_->descriptorPool, 0),
		            "vkResetDescriptorPool before dispatch");

		VkDescriptorSet descriptorSet{};
		const VkDescriptorSetAllocateInfo setInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = impl_->descriptorPool,
			.descriptorSetCount = 1,
			.pSetLayouts = &impl_->descriptorSetLayout,
		};
		CheckVulkan(vkAllocateDescriptorSets(impl_->context->device, &setInfo, &descriptorSet),
		            "vkAllocateDescriptorSets");

		std::vector<VkDescriptorBufferInfo> bufferInfos;
		bufferInfos.reserve(descriptorBuffers.size());
		std::vector<VkWriteDescriptorSet> writes;
		writes.reserve(descriptorBuffers.size());
		for (std::uint32_t i = 0; i < descriptorBuffers.size(); ++i)
		{
			const auto& buffer = AsBuffer(descriptorBuffers[i]);
			bufferInfos.push_back({
			    .buffer = buffer.buffer,
			    .offset = 0,
			    .range = buffer.byteSize,
			});
			writes.push_back({
			    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			    .dstSet = descriptorSet,
			    .dstBinding = i,
			    .descriptorCount = 1,
			    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			    .pBufferInfo = &bufferInfos.back(),
			});
		}
		vkUpdateDescriptorSets(impl_->context->device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
		                       nullptr);

		VkCommandBuffer commandBuffer{};
		const VkCommandBufferAllocateInfo commandInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = impl_->context->commandPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		CheckVulkan(vkAllocateCommandBuffers(impl_->context->device, &commandInfo, &commandBuffer),
		            "vkAllocateCommandBuffers");

		VulkanQueryPoolGuard timestampPool(impl_->context->device);
		if (recordGpuTimestamp)
		{
			const VkQueryPoolCreateInfo queryPoolInfo{
				.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
				.queryType = VK_QUERY_TYPE_TIMESTAMP,
				.queryCount = 2,
			};
			CheckVulkan(vkCreateQueryPool(impl_->context->device, &queryPoolInfo, nullptr,
			                              &timestampPool.queryPool),
			            "vkCreateQueryPool(timestamp)");
		}

		const VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		CheckVulkan(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
		if (recordGpuTimestamp)
		{
			vkCmdResetQueryPool(commandBuffer, timestampPool.queryPool, 0, 2);
			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool.queryPool, 0);
		}
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipelineLayout, 0, 1,
		                        &descriptorSet, 0, nullptr);
		vkCmdDispatch(commandBuffer, groups.x, groups.y, groups.z);
		if (recordGpuTimestamp)
		{
			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool.queryPool, 1);
		}
		CheckVulkan(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

		VkFence fence{};
		const VkFenceCreateInfo fenceInfo{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		CheckVulkan(vkCreateFence(impl_->context->device, &fenceInfo, nullptr, &fence), "vkCreateFence");
		const VkSubmitInfo submitInfo{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &commandBuffer,
		};
		const auto submitResult = vkQueueSubmit(impl_->context->queue, 1, &submitInfo, fence);
		if (submitResult == VK_SUCCESS)
		{
			CheckVulkan(vkWaitForFences(impl_->context->device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
			if (recordGpuTimestamp)
			{
				std::array<std::uint64_t, 2> timestamps{};
				const auto queryResult = vkGetQueryPoolResults(
				    impl_->context->device, timestampPool.queryPool, 0, 2, sizeof(timestamps), timestamps.data(),
				    sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
				if (queryResult == VK_SUCCESS)
				{
					const auto delta = VulkanTimestampDelta(timestamps[0], timestamps[1],
					                                        capabilities.computeQueueTimestampValidBits);
					options.timing->gpuTimestampAvailable = true;
					options.timing->gpuElapsedMs =
					    static_cast<double>(delta) * capabilities.timestampPeriodNanoseconds / 1'000'000.0;
				}
				else if (queryResult != VK_NOT_READY)
				{
					CheckVulkan(queryResult, "vkGetQueryPoolResults(timestamp)");
				}
			}
		}
		vkDestroyFence(impl_->context->device, fence, nullptr);
		vkFreeCommandBuffers(impl_->context->device, impl_->context->commandPool, 1, &commandBuffer);
		const auto resetResult = vkResetDescriptorPool(impl_->context->device, impl_->descriptorPool, 0);
		CheckVulkan(submitResult, "vkQueueSubmit");
		CheckVulkan(resetResult, "vkResetDescriptorPool after dispatch");
	}
} // namespace LiteNN

#endif
