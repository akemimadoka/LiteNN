#ifndef LITENN_TENSOR_H
#define LITENN_TENSOR_H

#include <LiteNN/Device.h>
#include <concepts>

namespace LiteNN
{
	struct UninitializedTag
	{
	};

	inline constexpr UninitializedTag Uninitialized{};

	template <Device D>
	class Tensor
	{
		constexpr static void ShapeSanCheck(ShapeView shape)
		{
			for (const auto dim : shape.Dims)
			{
				if (dim == 0)
				{
					throw std::runtime_error("Shape dimensions must be greater than 0");
				}
			}
		}

	public:
		using UsingDevice = D;

		constexpr Tensor(UninitializedTag, ShapeView shape, DataType dtype = DataType::Float32,
		                 UsingDevice device = UsingDevice())
		    : shape_(shape.Dims.begin(), shape.Dims.end()), strides_(ComputeContiguousStrides(shape)), dtype_(dtype),
		      device_(std::move(device)), ownsData_(true)
		{
			ShapeSanCheck(Shape());
			allocatedNumElements_ = NumElements();
			data_ = DeviceTraits<D>::Allocate(device_, dtype_, allocatedNumElements_);
		}

		constexpr Tensor(ShapeView shape, DataType dtype = DataType::Float32, UsingDevice device = UsingDevice())
		    : Tensor(Uninitialized, shape, dtype, std::move(device))
		{
			DeviceTraits<D>::ZeroFill(device_, data_, dtype_, allocatedNumElements_);
		}

		constexpr Tensor(std::span<const double> flatInitializer, ShapeView shape, DataType dtype = DataType::Float32,
		                 UsingDevice device = UsingDevice())
		    : Tensor(Uninitialized, shape, dtype, std::move(device))
		{
			assert(flatInitializer.size() == allocatedNumElements_);
			DeviceTraits<D>::CopyFromCPU(device_, dtype_, data_, DataType::Float64, flatInitializer.data(),
			                             allocatedNumElements_);
		}

		// GCC libstdc++ 尚未实现 P2447R6（span 的 initializer_list 构造），需要显式重载
		constexpr Tensor(std::initializer_list<double> flatInitializer, ShapeView shape,
		                 DataType dtype = DataType::Float32, UsingDevice device = UsingDevice())
		    : Tensor(std::span<const double>(flatInitializer), shape, dtype, std::move(device))
		{
		}

		// dtype 必须显式给出，避免漏填
		// allocatedNumElements_ 此时无意义
		constexpr Tensor(void* externalData, ShapeView shape, DataType dtype, UsingDevice device = UsingDevice())
		    : data_(externalData), allocatedNumElements_(), shape_(shape.Dims.begin(), shape.Dims.end()),
		      strides_(ComputeContiguousStrides(shape)), dtype_(dtype), device_(std::move(device)), ownsData_(false)
		{
			ShapeSanCheck(Shape());
		}

		constexpr ~Tensor()
		{
			if (ownsData_)
			{
				DeviceTraits<D>::Deallocate(device_, data_, dtype_, allocatedNumElements_);
			}
		}

		constexpr Tensor(const Tensor& other)
		    : data_(DeviceTraits<D>::Allocate(other.device_, other.dtype_, other.NumElements())),
		      allocatedNumElements_(other.allocatedNumElements_), shape_(other.shape_),
		      strides_(ComputeContiguousStrides(ShapeView{ other.shape_ })), dtype_(other.dtype_),
		      device_(other.device_), ownsData_(true)
		{
			DeviceTraits<D>::ConvertTo(device_, other.dtype_, other.data_, other.NumElements(), dtype_, data_);
		}

		constexpr Tensor(Tensor&& other) noexcept
		    : data_(other.data_), allocatedNumElements_(other.allocatedNumElements_), shape_(std::move(other.shape_)),
		      strides_(std::move(other.strides_)), dtype_(other.dtype_), device_(std::move(other.device_)),
		      ownsData_(other.ownsData_)
		{
			other.data_ = nullptr;
			other.ownsData_ = false;
		}

		// NOTE: 若当前 Tensor
		// 是一个视图，复制赋值操作不会分配新的内存，而是将值写入原数据的值到引用数据上
		// 这是为了使得 tensor[0] = tensor2 这样的操作符合直觉
		constexpr Tensor& operator=(const Tensor& other)
		{
			if (this != &other)
			{
				if (ownsData_)
				{
					const auto numElements = NumElements();
					DeviceTraits<D>::Deallocate(device_, data_, dtype_, numElements);
					shape_ = other.shape_;
					strides_ = ComputeContiguousStrides(Shape());
					dtype_ = other.dtype_;
					device_ = other.device_;
					allocatedNumElements_ = other.allocatedNumElements_;
					data_ = DeviceTraits<D>::Allocate(device_, dtype_, other.NumElements());
					DeviceTraits<D>::ConvertTo(device_, other.dtype_, other.data_, other.NumElements(), dtype_, data_);
				}
				else
				{
					if (shape_ != other.shape_ || dtype_ != other.dtype_)
					{
						throw std::runtime_error("Cannot assign to a view tensor with different shape or data type");
					}
					DeviceTraits<D>::ConvertTo(device_, other.dtype_, other.data_, other.NumElements(), dtype_, data_);
				}
			}
			return *this;
		}

		// NOTE: 若当前 Tensor 是一个视图，移动赋值操作不是替换当前 Tensor
		// 的数据，而是写入原数据的值到引用数据上
		// 这是为了使得 tensor[0] = tensor2 + tensor3 这样的操作符合直觉
		// TODO: 这会使得移动赋值不能是 noexcept 的
		constexpr Tensor& operator=(Tensor&& other) noexcept(false)
		{
			if (this != &other)
			{
				if (ownsData_)
				{
					const auto numElements = NumElements();
					DeviceTraits<D>::Deallocate(device_, data_, dtype_, numElements);
					shape_ = std::move(other.shape_);
					strides_ = std::move(other.strides_);
					dtype_ = other.dtype_;
					device_ = std::move(other.device_);
					ownsData_ = other.ownsData_;
					data_ = other.data_;
					allocatedNumElements_ = other.allocatedNumElements_;

					other.data_ = nullptr;
					other.ownsData_ = false;
				}
				else
				{
					if (shape_ != other.shape_ || dtype_ != other.dtype_)
					{
						throw std::runtime_error("Cannot assign to a view tensor with different shape or data type");
					}
					DeviceTraits<D>::ConvertTo(device_, other.dtype_, other.data_, other.NumElements(), dtype_, data_);
				}
			}
			return *this;
		}

		constexpr auto& CurDevice(this auto&& self)
		{
			return self.device_;
		}

		constexpr ShapeView Shape() const
		{
			return { shape_ };
		}

		constexpr bool IsScalar() const
		{
			return Shape().IsScalar();
		}

		constexpr std::span<const std::size_t> Strides() const
		{
			return strides_;
		}

		constexpr bool IsContiguous() const
		{
			return strides_ == ComputeContiguousStrides(Shape());
		}

		// TODO: 实现 stride，允许不连续的内存布局
		constexpr void Reshape(ShapeView newShape)
		{
			if (!IsContiguous())
			{
				throw std::runtime_error("Reshape requires contiguous tensor");
			}
			const auto newNumElements = newShape.NumElements();
			if (newNumElements != NumElements())
			{
				throw std::runtime_error("Reshape failed: total number of elements does not match");
			}
			shape_.assign(newShape.Dims.begin(), newShape.Dims.end());
			strides_ = ComputeContiguousStrides(Shape());
		}

		constexpr std::size_t NumElements() const
		{
			return Shape().NumElements();
		}

		constexpr std::size_t AllocatedNumElements() const
		{
			return allocatedNumElements_;
		}

		constexpr DataType DType() const
		{
			return dtype_;
		}

		constexpr void* RawData()
		{
			return data_;
		}

		constexpr const void* RawData() const
		{
			return data_;
		}

		// SAFETY: 这个操作会返回一个新的 Tensor
		// 视图，使用者需保证使用时，返回值的生命周期不超过当前 Tensor 的生命周期
		constexpr Tensor operator[](std::convertible_to<std::size_t> auto... index)
		{
			const auto shape = Shape();
			assert(shape.NumDim() > 0);
			assert(shape.NumDim() >= sizeof...(index));
			auto newShape = shape;
			auto newData = data_;
			std::size_t dimIndex = 0;
			template for (const std::size_t i : { index... })
			{
				if (i >= newShape[0])
				{
					throw std::out_of_range("Index out of bounds");
				}
				const auto currentStride = strides_[dimIndex];
				newShape = newShape.SubShape(1);
				newData = EnumDispatch(dtype_, [&]<DataType TypeValue> {
					using T = typename DeviceTraits<D>::template DataTypeMapping<TypeValue>;
					const auto offset = i * currentStride * sizeof(T);
					return static_cast<void*>(static_cast<char*>(newData) + offset);
				});
				++dimIndex;
			}
			// TODO: 在 Tensor 视图中，Device 是否需要复制？
			return Tensor(newData, newShape, dtype_, device_);
		}

		constexpr Tensor operator-() const
		{
			Tensor result(Shape(), DType(), device_);
			DeviceTraits<D>::DoUnaryOp(device_, UnaryOp::Negate, result.RawData(), dtype_, Shape(), data_);
			return result;
		}

		constexpr Tensor Abs() const
		{
			Tensor result(Shape(), DType(), device_);
			DeviceTraits<D>::DoUnaryOp(device_, UnaryOp::Abs, result.RawData(), dtype_, Shape(), data_);
			return result;
		}

		constexpr Tensor Sqrt() const
		{
			Tensor result(Shape(), DType(), device_);
			DeviceTraits<D>::DoUnaryOp(device_, UnaryOp::Sqrt, result.RawData(), dtype_, Shape(), data_);
			return result;
		}

		constexpr Tensor Transpose() const
		{
			const auto resultShape = UnaryOpTraits<UnaryOp::Transpose>::ResultShape(Shape());
			const auto resultType = UnaryOpTraits<UnaryOp::Transpose>::ResultType(DType());
			assert(resultShape && resultType);
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoUnaryOp(device_, UnaryOp::Transpose, result.RawData(), dtype_, Shape(), data_);
			return result;
		}

		constexpr Tensor operator+(const Tensor& other) const
		{
			const auto resultShape = BinaryOpTraits<BinaryOp::Add>::ResultShape(Shape(), other.Shape());
			const auto resultType = BinaryOpTraits<BinaryOp::Add>::ResultType(DType(), other.DType());
			if (!resultShape)
			{
				throw std::runtime_error(resultShape.error());
			}
			if (!resultType)
			{
				throw std::runtime_error(resultType.error());
			}
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoBinaryOp(device_, BinaryOp::Add, result.RawData(), dtype_, Shape(), data_, other.dtype_,
			                            other.Shape(), other.data_);
			return result;
		}

		constexpr Tensor operator-(const Tensor& other) const
		{
			const auto resultShape = BinaryOpTraits<BinaryOp::Subtract>::ResultShape(Shape(), other.Shape());
			const auto resultType = BinaryOpTraits<BinaryOp::Subtract>::ResultType(DType(), other.DType());
			if (!resultShape)
			{
				throw std::runtime_error(resultShape.error());
			}
			if (!resultType)
			{
				throw std::runtime_error(resultType.error());
			}
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoBinaryOp(device_, BinaryOp::Subtract, result.RawData(), dtype_, Shape(), data_,
			                            other.dtype_, other.Shape(), other.data_);
			return result;
		}

		// NOTE: hadamard 积
		constexpr Tensor operator*(const Tensor& other) const
		{
			const auto resultShape = BinaryOpTraits<BinaryOp::Multiply>::ResultShape(Shape(), other.Shape());
			const auto resultType = BinaryOpTraits<BinaryOp::Multiply>::ResultType(DType(), other.DType());
			if (!resultShape)
			{
				throw std::runtime_error(resultShape.error());
			}
			if (!resultType)
			{
				throw std::runtime_error(resultType.error());
			}
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoBinaryOp(device_, BinaryOp::Multiply, result.RawData(), dtype_, Shape(), data_,
			                            other.dtype_, other.Shape(), other.data_);
			return result;
		}

		constexpr Tensor operator/(const Tensor& other) const
		{
			const auto resultShape = BinaryOpTraits<BinaryOp::Divide>::ResultShape(Shape(), other.Shape());
			const auto resultType = BinaryOpTraits<BinaryOp::Divide>::ResultType(DType(), other.DType());
			if (!resultShape)
			{
				throw std::runtime_error(resultShape.error());
			}
			if (!resultType)
			{
				throw std::runtime_error(resultType.error());
			}
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoBinaryOp(device_, BinaryOp::Divide, result.RawData(), dtype_, Shape(), data_,
			                            other.dtype_, other.Shape(), other.data_);
			return result;
		}

		constexpr Tensor MatMul(const Tensor& other) const
		{
			const auto resultShape = BinaryOpTraits<BinaryOp::MatMul>::ResultShape(Shape(), other.Shape());
			const auto resultType = BinaryOpTraits<BinaryOp::MatMul>::ResultType(DType(), other.DType());
			if (!resultShape)
			{
				throw std::runtime_error(resultShape.error());
			}
			if (!resultType)
			{
				throw std::runtime_error(resultType.error());
			}
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoBinaryOp(device_, BinaryOp::MatMul, result.RawData(), dtype_, Shape(), data_,
			                            other.dtype_, other.Shape(), other.data_);
			return result;
		}

		constexpr Tensor operator<(const Tensor& other) const
		{
			const auto resultShape = BinaryOpTraits<BinaryOp::Less>::ResultShape(Shape(), other.Shape());
			const auto resultType = BinaryOpTraits<BinaryOp::Less>::ResultType(DType(), other.DType());
			if (!resultShape)
			{
				throw std::runtime_error(resultShape.error());
			}
			if (!resultType)
			{
				throw std::runtime_error(resultType.error());
			}
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoBinaryOp(device_, BinaryOp::Less, result.RawData(), dtype_, Shape(), data_, other.dtype_,
			                            other.Shape(), other.data_);
			return result;
		}

		constexpr Tensor operator>(const Tensor& other) const
		{
			const auto resultShape = BinaryOpTraits<BinaryOp::Greater>::ResultShape(Shape(), other.Shape());
			const auto resultType = BinaryOpTraits<BinaryOp::Greater>::ResultType(DType(), other.DType());
			if (!resultShape)
			{
				throw std::runtime_error(resultShape.error());
			}
			if (!resultType)
			{
				throw std::runtime_error(resultType.error());
			}
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoBinaryOp(device_, BinaryOp::Greater, result.RawData(), dtype_, Shape(), data_,
			                            other.dtype_, other.Shape(), other.data_);
			return result;
		}

		constexpr Tensor operator<=(const Tensor& other) const
		{
			return !(*this > other);
		}

		constexpr Tensor operator>=(const Tensor& other) const
		{
			return !(*this < other);
		}

		constexpr Tensor operator==(const Tensor& other) const
		{
			const auto resultShape = BinaryOpTraits<BinaryOp::Equal>::ResultShape(Shape(), other.Shape());
			const auto resultType = BinaryOpTraits<BinaryOp::Equal>::ResultType(DType(), other.DType());
			if (!resultShape)
			{
				throw std::runtime_error(resultShape.error());
			}
			if (!resultType)
			{
				throw std::runtime_error(resultType.error());
			}
			Tensor result(Uninitialized, *resultShape, *resultType, device_);
			DeviceTraits<D>::DoBinaryOp(device_, BinaryOp::Equal, result.RawData(), dtype_, Shape(), data_,
			                            other.dtype_, other.Shape(), other.data_);
			return result;
		}

		constexpr Tensor operator!() const
		{
			Tensor result(Shape(), DType(), device_);
			DeviceTraits<D>::DoUnaryOp(device_, UnaryOp::LogicalNegation, result.RawData(), dtype_, Shape(), data_);
			return result;
		}

		template <Device OtherDevice>
		constexpr Tensor<OtherDevice> CopyToDevice(OtherDevice otherDevice) const
		{
			Tensor<OtherDevice> result(Shape(), DType(), std::move(otherDevice));
			if constexpr (std::same_as<D, PolymorphicDevice>)
			{
				if constexpr (std::same_as<OtherDevice, PolymorphicDevice>)
				{
					if (device_.IsSameDevice(result.CurDevice()))
					{
						DeviceTraits<D>::ConvertTo(device_, dtype_, data_, NumElements(), dtype_, result.RawData());
					}
					else
					{
						// 设备不同，先复制到 CPU，再从 CPU 复制到目标设备
						Tensor<CPU> temp(Uninitialized, Shape(), DType(), CPU{});
						DeviceTraits<D>::CopyToCPU(device_, dtype_, data_, NumElements(), dtype_, temp.RawData());
						DeviceTraits<OtherDevice>::CopyFromCPU(result.CurDevice(), dtype_, result.RawData(), dtype_,
						                                       temp.RawData(), NumElements());
					}
				}
				else if (device_.template Is<OtherDevice>())
				{
					DeviceTraits<D>::ConvertTo(device_, dtype_, data_, NumElements(), dtype_, result.RawData());
				}
				else
				{
					// 设备不同，先复制到 CPU，再从 CPU 复制到目标设备
					Tensor<CPU> temp(Uninitialized, Shape(), DType(), CPU{});
					DeviceTraits<D>::CopyToCPU(device_, dtype_, data_, NumElements(), dtype_, temp.RawData());
					DeviceTraits<OtherDevice>::CopyFromCPU(result.CurDevice(), dtype_, result.RawData(), dtype_,
					                                       temp.RawData(), NumElements());
				}
			}
			else if constexpr (std::same_as<OtherDevice, PolymorphicDevice>)
			{
				if (result.CurDevice().template Is<D>())
				{
					DeviceTraits<D>::ConvertTo(device_, dtype_, data_, NumElements(), dtype_, result.RawData());
				}
				else
				{
					// 设备不同，先复制到 CPU，再从 CPU 复制到目标设备
					Tensor<CPU> temp(Uninitialized, Shape(), DType(), CPU{});
					DeviceTraits<D>::CopyToCPU(device_, dtype_, data_, NumElements(), dtype_, temp.RawData());
					DeviceTraits<OtherDevice>::CopyFromCPU(result.CurDevice(), dtype_, result.RawData(), dtype_,
					                                       temp.RawData(), NumElements());
				}
			}
			else if constexpr (std::same_as<D, OtherDevice>)
			{
				DeviceTraits<D>::ConvertTo(device_, dtype_, data_, NumElements(), dtype_, result.RawData());
			}
			else if constexpr (std::same_as<OtherDevice, CPU>)
			{
				DeviceTraits<D>::CopyToCPU(device_, dtype_, data_, NumElements(), dtype_, result.RawData());
			}
			else if constexpr (std::same_as<D, CPU>)
			{
				DeviceTraits<OtherDevice>::CopyFromCPU(result.CurDevice(), dtype_, result.RawData(), dtype_, data_,
				                                       NumElements());
			}
			else
			{
				// 其他设备之间的复制，先复制到 CPU，再从 CPU 复制到目标设备
				Tensor<CPU> temp(Uninitialized, Shape(), DType(), CPU{});
				DeviceTraits<D>::CopyToCPU(device_, dtype_, data_, NumElements(), dtype_, temp.RawData());
				DeviceTraits<OtherDevice>::CopyFromCPU(result.CurDevice(), dtype_, result.RawData(), dtype_,
				                                       temp.RawData(), NumElements());
			}
			return result;
		}

	private:
		void* data_;
		std::size_t allocatedNumElements_; // Reshape 后 numElements 将不会等同于 shape 的
		                                   // numElements，此值用于释放存储时正确计算大小
		bool ownsData_;                    // 为 false 时，表示是另一个 Tensor
		                                   // 的视图，不拥有数据所有权，不负责内存管理
		[[no_unique_address]] mutable UsingDevice device_;
		std::vector<std::size_t> shape_;
		std::vector<std::size_t> strides_;
		DataType dtype_;

		static std::vector<std::size_t> ComputeContiguousStrides(ShapeView shape)
		{
			std::vector<std::size_t> strides(shape.NumDim());
			if (!strides.empty())
			{
				strides.back() = 1;
				for (auto i = shape.NumDim() - 1; i > 0; --i)
				{
					strides[i - 1] = strides[i] * shape[i];
				}
			}
			return strides;
		}
	};
} // namespace LiteNN

template <LiteNN::Device D>
struct std::formatter<LiteNN::Tensor<D>> : std::formatter<std::string_view>
{
	template <class ParseContext>
	constexpr ParseContext::iterator parse(ParseContext& ctx)
	{
		return ctx.begin();
	}

	template <class FormatContext>
	auto format(const LiteNN::Tensor<D>& tensor, FormatContext& ctx) const
	{
		using namespace LiteNN;

		// numpy/torch 风格输出格式
		const auto deviceName = DeviceTraits<D>::Name();
		const auto deviceInfo = DeviceTraits<D>::Info(tensor.CurDevice());
		const auto dtypeName = EnumToString<EnumToStringStyle::Unqualified>(tensor.DType());
		// 如果不是 CPU 设备，尝试将 tensor 复制到 CPU 上以便打印内容
		std::optional<Tensor<CPU>> cpuTensor;
		const auto& tempTensor = [&] -> const Tensor<CPU>& {
			if constexpr (std::same_as<D, CPU>)
			{
				return tensor;
			}
			else
			{
				cpuTensor.emplace(tensor.CopyToDevice(CPU{}));
				return *cpuTensor;
			}
		}();
		auto outIt = ctx.out();
		outIt = std::ranges::copy("Tensor(", outIt).out;
		EnumDispatch(tempTensor.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			if (tempTensor.Shape().NumDim() == 0)
			{
				// 标量特殊处理，直接输出值
				outIt = std::format_to(outIt, "{}", *static_cast<const T*>(tempTensor.RawData()));
				return;
			}
			const auto elementPrinter = [&](this auto&& self, const T* data, ShapeView shape) -> void {
				outIt = std::ranges::copy("[ ", outIt).out;
				if (shape.NumDim() == 1)
				{
					const auto totalElements = shape.NumElements();
					// Shape 已检查过，不可能包含 0，因此至少有 1 个元素
					outIt = std::format_to(outIt, "{}", data[0]);
					for (auto i = 1uz; i < totalElements; ++i)
					{
						outIt = std::format_to(outIt, ", {}", data[i]);
					}
				}
				else
				{
					const auto subShape = shape.SubShape(1);
					const auto subShapeNumElements = subShape.NumElements();
					// Shape 已检查过，不可能包含 0，因此至少有 1 个元素
					self(data, subShape);
					for (auto i = 1uz; i < shape[0]; ++i)
					{
						outIt = std::ranges::copy(", ", outIt).out;
						self(data + i * subShapeNumElements, subShape);
					}
				}
				outIt = std::ranges::copy(" ]", outIt).out;
			};
			elementPrinter(static_cast<const T*>(tempTensor.RawData()), tempTensor.Shape());
		});
		return std::format_to(outIt, ", dtype={}, device=\"{}({})\")", dtypeName, deviceName, deviceInfo);
	}
};

#endif
