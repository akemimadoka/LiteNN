#ifndef LITENN_DEVICE_H
#define LITENN_DEVICE_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <memory>

#include <LiteNN/Operators.h>

namespace LiteNN
{
	template <typename T>
	struct DeviceTraits;

	template <typename T>
	concept Device =
	    requires(T& device, const T& constDevice, DataType type, std::size_t size, ShapeView shape, void* ptr,
	             void* dst, const void* src1, const void* src2, UnaryOp unaryOp, BinaryOp binaryOp, ReduceOp reduceOp) {
		    // TODO: 无法过编译，猜测是因为无法表达 type 是一个 constexpr 参数的问题
		    // typename DeviceTraits<T>::template DataTypeMapping<type>;

		    { DeviceTraits<T>::Name() } -> std::same_as<std::string_view>;
		    { DeviceTraits<T>::Info(constDevice) } -> std::same_as<std::string_view>;

		    // 基础分配操作
		    // 分配的存储默认未初始化
		    // size 参数表示分配的元素数量（而非字节大小），调用者需要根据 type
		    // 来计算实际的字节大小
		    { DeviceTraits<T>::Allocate(device, type, size) } -> std::same_as<void*>;
		    { DeviceTraits<T>::Deallocate(device, ptr, type, size) } -> std::same_as<void>;
		    { DeviceTraits<T>::ZeroFill(device, ptr, type, size) } -> std::same_as<void>;
		    // TODO: 当前没有考虑 stride 的情况，后续可以增加带 stride
		    // 参数的版本，现在需要调用者自己重复调用多次来实现
		    { DeviceTraits<T>::CopyToCPU(device, type, src1, size, type, dst) } -> std::same_as<void>;
		    { DeviceTraits<T>::CopyFromCPU(device, type, dst, type, src1, size) } -> std::same_as<void>;
		    { DeviceTraits<T>::ConvertTo(device, type, src1, size, type, dst) } -> std::same_as<void>;

		    // TODO: 分离到 DeviceExecutor
		    // dst = unaryOp(src)
		    // dst 需要预先分配好内存，且类型和 shape 由 UnaryOpTraits 决定
		    { DeviceTraits<T>::DoUnaryOp(device, unaryOp, dst, type, shape, src1) } -> std::same_as<void>;
		    // dst = binaryOp(src1, src2)
		    // dst 需要预先分配好内存，且类型和 shape 由 BinaryOpTraits 决定
		    {
			    DeviceTraits<T>::DoBinaryOp(device, binaryOp, dst, type, shape, src1, type, shape, src2)
		    } -> std::same_as<void>;
		    // dst = reduceOp(src, axis)
		    // dst 需要预先分配好内存，且类型和 shape 由 ReduceOpTraits 决定
		    { DeviceTraits<T>::DoReduceOp(device, reduceOp, dst, type, shape, src1, size) } -> std::same_as<void>;
		    // dst = concat(srcs..., axis)
		    // srcPtrs 和 srcShapes 分别是各输入数据指针和 shape 的数组
		    {
			    DeviceTraits<T>::DoConcatOp(device, dst, type, (const void* const*)nullptr,
			                               (const ShapeView*)nullptr, size, size)
		    } -> std::same_as<void>;
		    // dst = slice(src, axis, start, length)
		    { DeviceTraits<T>::DoSliceOp(device, dst, type, shape, src1, size, size, size) } -> std::same_as<void>;
	    };

	// 擦除了类型的 Device
	// TODO: 无法实现 DataTypeMapping，考虑拆分到返回类型特征的函数里
	class PolymorphicDevice
	{
	public:
		PolymorphicDevice(Device auto device) : impl_(std::make_unique<Impl<decltype(device)>>(std::move(device)))
		{
		}

		PolymorphicDevice(const PolymorphicDevice& other) : impl_(other.impl_->Clone())
		{
		}

		template <Device D>
		bool Is() const
		{
			return dynamic_cast<Impl<D>*>(impl_.get()) != nullptr;
		}

		template <Device D>
		D* As()
		{
			if (const auto impl = dynamic_cast<Impl<D>*>(impl_.get()))
			{
				return std::addressof(impl->device_);
			}
			else
			{
				return nullptr;
			}
		}

		bool IsSameDeviceType(const PolymorphicDevice& other) const
		{
			return typeid(*impl_) == typeid(*other.impl_);
		}

		bool IsSameDevice(const PolymorphicDevice& other) const
		{
			return impl_->IsSameDevice(*other.impl_);
		}

	private:
		struct Interface
		{
			virtual ~Interface();
			virtual std::unique_ptr<Interface> Clone() const = 0;
			virtual std::string_view Name() const = 0;
			virtual std::string_view Info() const = 0;
			virtual void* Allocate(DataType type, std::size_t size) = 0;
			virtual void Deallocate(void* ptr, DataType type, std::size_t size) = 0;
			virtual void ZeroFill(void* ptr, DataType type, std::size_t size) = 0;
			virtual void CopyToCPU(DataType srcType, const void* src, std::size_t size, DataType dstType,
			                       void* dst) = 0;
			virtual void CopyFromCPU(DataType dstType, void* dst, DataType srcType, const void* src,
			                         std::size_t size) = 0;
			virtual void ConvertTo(DataType srcType, const void* src, std::size_t size, DataType dstType,
			                       void* dst) = 0;
			virtual void DoUnaryOp(UnaryOp unaryOp, void* dst, DataType type, ShapeView shape, const void* src) = 0;
			virtual void DoBinaryOp(BinaryOp binaryOp, void* dst, DataType type1, ShapeView shape1, const void* src1,
			                        DataType type2, ShapeView shape2, const void* src2) = 0;
			virtual void DoReduceOp(ReduceOp reduceOp, void* dst, DataType type, ShapeView shape, const void* src,
			                        std::size_t axis) = 0;
			virtual void DoConcatOp(void* dst, DataType type, const void* const* srcPtrs,
			                        const ShapeView* srcShapes, std::size_t inputCount, std::size_t axis) = 0;
			virtual void DoSliceOp(void* dst, DataType type, ShapeView srcShape, const void* src,
			                       std::size_t axis, std::size_t start, std::size_t length) = 0;
			virtual bool IsSameDevice(const Interface& other) const = 0;
		};

		template <Device D>
		class Impl : public Interface
		{
		public:
			Impl(D device) : device_(std::move(device))
			{
			}
			~Impl() override = default;

			std::unique_ptr<Interface> Clone() const override
			{
				return std::make_unique<Impl<D>>(device_);
			}

			std::string_view Name() const override
			{
				return DeviceTraits<D>::Name();
			}
			std::string_view Info() const override
			{
				return DeviceTraits<D>::Info(device_);
			}
			void* Allocate(DataType type, std::size_t size) override
			{
				return DeviceTraits<D>::Allocate(device_, type, size);
			}
			void Deallocate(void* ptr, DataType type, std::size_t size) override
			{
				DeviceTraits<D>::Deallocate(device_, ptr, type, size);
			}
			void ZeroFill(void* ptr, DataType type, std::size_t size) override
			{
				DeviceTraits<D>::ZeroFill(device_, ptr, type, size);
			}
			void CopyToCPU(DataType srcType, const void* src, std::size_t size, DataType dstType, void* dst) override
			{
				DeviceTraits<D>::CopyToCPU(device_, srcType, src, size, dstType, dst);
			}
			void CopyFromCPU(DataType dstType, void* dst, DataType srcType, const void* src, std::size_t size) override
			{
				DeviceTraits<D>::CopyFromCPU(device_, dstType, dst, srcType, src, size);
			}
			void ConvertTo(DataType srcType, const void* src, std::size_t size, DataType dstType, void* dst) override
			{
				DeviceTraits<D>::ConvertTo(device_, srcType, src, size, dstType, dst);
			}
			void DoUnaryOp(UnaryOp unaryOp, void* dst, DataType type, ShapeView shape, const void* src) override
			{
				DeviceTraits<D>::DoUnaryOp(device_, unaryOp, dst, type, shape, src);
			}
			void DoBinaryOp(BinaryOp binaryOp, void* dst, DataType type1, ShapeView shape1, const void* src1,
			                DataType type2, ShapeView shape2, const void* src2) override
			{
				DeviceTraits<D>::DoBinaryOp(device_, binaryOp, dst, type1, shape1, src1, type2, shape2, src2);
			}
			void DoReduceOp(ReduceOp reduceOp, void* dst, DataType type, ShapeView shape, const void* src,
			                std::size_t axis) override
			{
				DeviceTraits<D>::DoReduceOp(device_, reduceOp, dst, type, shape, src, axis);
			}
			void DoConcatOp(void* dst, DataType type, const void* const* srcPtrs,
			                const ShapeView* srcShapes, std::size_t inputCount, std::size_t axis) override
			{
				DeviceTraits<D>::DoConcatOp(device_, dst, type, srcPtrs, srcShapes, inputCount, axis);
			}
			void DoSliceOp(void* dst, DataType type, ShapeView srcShape, const void* src,
			               std::size_t axis, std::size_t start, std::size_t length) override
			{
				DeviceTraits<D>::DoSliceOp(device_, dst, type, srcShape, src, axis, start, length);
			}
			bool IsSameDevice(const Interface& other) const override
			{
				if (auto* p = dynamic_cast<const Impl<D>*>(&other))
				{
					return device_ == p->device_;
				}
				return false;
			}

		private:
			D device_;
		};

		std::unique_ptr<Interface> impl_;
		mutable std::string infoCache_;

		friend struct DeviceTraits<PolymorphicDevice>;
	};

	template <>
	struct DeviceTraits<PolymorphicDevice>
	{
		static std::string_view Name();
		static std::string_view Info(const PolymorphicDevice& device);
		static void* Allocate(PolymorphicDevice& device, DataType type, std::size_t size);
		static void Deallocate(PolymorphicDevice& device, void* ptr, DataType type, std::size_t size);
		static void ZeroFill(PolymorphicDevice& device, void* ptr, DataType type, std::size_t size);
		static void CopyToCPU(PolymorphicDevice& device, DataType srcType, const void* src, std::size_t size,
		                      DataType dstType, void* dst);
		static void CopyFromCPU(PolymorphicDevice& device, DataType dstType, void* dst, DataType srcType,
		                        const void* src, std::size_t size);
		static void ConvertTo(PolymorphicDevice& device, DataType srcType, const void* src, std::size_t size,
		                      DataType dstType, void* dst);
		static void DoUnaryOp(PolymorphicDevice& device, UnaryOp unaryOp, void* dst, DataType type, ShapeView shape,
		                      const void* src);
		static void DoBinaryOp(PolymorphicDevice& device, BinaryOp binaryOp, void* dst, DataType type1,
		                       ShapeView shape1, const void* src1, DataType type2, ShapeView shape2, const void* src2);
		static void DoReduceOp(PolymorphicDevice& device, ReduceOp reduceOp, void* dst, DataType type, ShapeView shape,
		                       const void* src, std::size_t axis);
		static void DoConcatOp(PolymorphicDevice& device, void* dst, DataType type, const void* const* srcPtrs,
		                       const ShapeView* srcShapes, std::size_t inputCount, std::size_t axis);
		static void DoSliceOp(PolymorphicDevice& device, void* dst, DataType type, ShapeView srcShape, const void* src,
		                      std::size_t axis, std::size_t start, std::size_t length);
	};

	struct CPU
	{
		bool operator==(const CPU&) const = default;
	};

	template <>
	struct DeviceTraits<CPU>
	{
		static consteval std::meta::info DataTypeMappingFunc(DataType dataType)
		{
			switch (dataType)
			{
			case DataType::Float32:
				return ^^float;
			case DataType::Float64:
				return ^^double;
			case DataType::Int32:
				return ^^int32_t;
			case DataType::Int64:
				return ^^int64_t;
			case DataType::Bool:
				return ^^bool;
			}
		}

		template <DataType DT>
		using DataTypeMapping = [:DataTypeMappingFunc(DT):];

		static constexpr std::string_view Name()
		{
			return "CPU";
		}

		static constexpr std::string_view Info(const CPU& device)
		{
			return "Generic CPU Device";
		}

		static constexpr void* Allocate(CPU& device, DataType type, std::size_t size)
		{
			const auto typeSize = EnumDispatch(type, []<DataType type> { return sizeof(DataTypeMapping<type>); });
			return ::operator new(size * typeSize);
		}

		static constexpr void Deallocate(CPU& device, void* ptr, DataType type, std::size_t size)
		{
			::operator delete(ptr);
		}

		static constexpr void ZeroFill(CPU& device, void* ptr, DataType type, std::size_t size)
		{
			EnumDispatch(type, [&]<DataType type> {
				using T = DataTypeMapping<type>;
				std::ranges::uninitialized_fill(std::span(static_cast<T*>(ptr), size), T{});
			});
		}

		// TODO: stride
		static constexpr void CopyToCPU(CPU& device, DataType srcType, const void* src, std::size_t size,
		                                DataType dstType, void* dst)
		{
			// 在 CPU 设备上，CopyToCPU 和 ConvertTo 的实现相同
			ConvertTo(device, srcType, src, size, dstType, dst);
		}

		static constexpr void CopyFromCPU(CPU& device, DataType dstType, void* dst, DataType srcType, const void* src,
		                                  std::size_t size)
		{
			// 在 CPU 设备上，CopyFromCPU 和 CopyToCPU 的实现相同，仅参数顺序不同
			CopyToCPU(device, srcType, src, size, dstType, dst);
		}

		static constexpr void ConvertTo(CPU& device, DataType srcType, const void* src, std::size_t size,
		                                DataType dstType, void* dst)
		{
			EnumDispatch(srcType, [&]<DataType SrcTypeValue> {
				using SrcT = DataTypeMapping<SrcTypeValue>;
				if (srcType == dstType)
				{
					std::ranges::copy(std::span(static_cast<const std::byte*>(src), size * sizeof(SrcT)),
					                  static_cast<std::byte*>(dst));
					return;
				}
				EnumDispatch(dstType, [&]<DataType DstTypeValue> {
					using DstT = DataTypeMapping<DstTypeValue>;
					for (auto i = 0uz; i < size; ++i)
					{
						static_cast<DstT*>(dst)[i] = static_cast<DstT>(static_cast<const SrcT*>(src)[i]);
					}
				});
			});
		}

		static constexpr void DoUnaryOp(CPU& device, UnaryOp unaryOp, void* dst, DataType type, ShapeView shape,
		                                const void* src)
		{
			EnumDispatch(type, [&]<DataType TypeValue> {
				using T = DataTypeMapping<TypeValue>;
				EnumDispatch(unaryOp, [&]<UnaryOp OpValue> {
					using OpTrait = UnaryOpTraits<OpValue>;
					constexpr auto resultTypeOrError = OpTrait::ResultType(TypeValue);
					if constexpr (!resultTypeOrError)
					{
						throw std::runtime_error(resultTypeOrError.error());
					}
					else
					{
						using ResultType = DataTypeMapping<resultTypeOrError.value()>;
						// 根据 OpTrait::ResultShape(shape) 分配结果内存
						// 这里假设 dst 已经分配好足够的内存
						switch (OpValue)
						{
						case UnaryOp::Negate: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = -static_cast<const T*>(src)[i];
							}
							break;
						}
						case UnaryOp::Abs: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::abs(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Sqrt: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::sqrt(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Exp: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::exp(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Log: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::log(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Sin: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::sin(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Cos: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::cos(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Tan: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::tan(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Arcsin: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::asin(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Arccos: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::acos(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Arctan: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = std::atan(static_cast<const T*>(src)[i]);
							}
							break;
						}
						case UnaryOp::Transpose:
							// TODO: 此操作不适用 dst 和 src 内存重叠的情况，目前先假设它们不重叠
							assert(shape.NumDim() == 2);
							for (auto i = 0uz; i < shape[0]; ++i)
							{
								for (auto j = 0uz; j < shape[1]; ++j)
								{
									static_cast<ResultType*>(dst)[j * shape[0] + i] =
									    static_cast<const T*>(src)[i * shape[1] + j];
								}
							}
							break;
						case UnaryOp::LogicalNegation: {
							const auto totalElements = shape.NumElements();
							for (auto i = 0uz; i < totalElements; ++i)
							{
								static_cast<ResultType*>(dst)[i] = !static_cast<const T*>(src)[i];
							}
							break;
						}
						}
					}
				});
			});
		}

		static constexpr void DoBinaryOp(CPU& device, BinaryOp binaryOp, void* dst, DataType type1, ShapeView shape1,
		                                 const void* src1, DataType type2, ShapeView shape2, const void* src2)
		{
			EnumDispatch(binaryOp, [&]<BinaryOp OpValue> {
				using OpTrait = BinaryOpTraits<OpValue>;
				EnumDispatch(type1, [&]<DataType TypeValue1> {
					using T1 = DataTypeMapping<TypeValue1>;
					EnumDispatch(type2, [&]<DataType TypeValue2> {
						using T2 = DataTypeMapping<TypeValue2>;
						constexpr auto resultTypeOrError = OpTrait::ResultType(TypeValue1, TypeValue2);
						if constexpr (!resultTypeOrError)
						{
							throw std::runtime_error(resultTypeOrError.error());
						}
						else
						{
							constexpr auto resultTypeValue = *resultTypeOrError;
							using ResultType = DataTypeMapping<resultTypeValue>;
							const auto resultShapeOrError = OpTrait::ResultShape(shape1, shape2);
							if (!resultShapeOrError)
							{
								throw std::runtime_error(resultShapeOrError.error());
							}
							const auto resultShape = ShapeView{ *resultShapeOrError };
							const auto totalElements = resultShape.NumElements();
							// 此时 op 接受 2 个参数(T1, T2)，返回 1 个值(ResultType)
							const auto broadcastOp = [&](this auto&& self, ShapeView resultShape, void* dst,
							                             ShapeView shape1, const void* src1, ShapeView shape2,
							                             const void* src2, auto&& op) -> void {
								// 模拟 numpy/pytorch 的广播机制
								// 广播时，较小的维度如果是 1，或者不存在该维度，则在计算时重复使用该维度的值
								if (resultShape.NumDim() == 1)
								{
									// 最后一个维度，直接计算
									const auto dim1 = shape1[0];
									const auto dim2 = shape2[0];
									for (auto j = 0uz; j < resultShape[0]; ++j)
									{
										const auto& val1 = static_cast<const T1*>(src1)[(dim1 == 1 ? 0 : j)];
										const auto& val2 = static_cast<const T2*>(src2)[(dim2 == 1 ? 0 : j)];
										static_cast<ResultType*>(dst)[j] = op(val1, val2);
									}
								}
								else
								{
									const auto subShape = resultShape.SubShape(1);
									// 还有更高维度，需要递归处理
									for (auto i = 0uz; i < resultShape[0]; ++i)
									{
										self(subShape,
										     static_cast<std::byte*>(dst) +
										         i * subShape.NumElements() * sizeof(ResultType),
										     shape1.NumDim() > 1 ? shape1.SubShape(1) : shape1,
										     shape1.NumDim() > 1
										         ? static_cast<const std::byte*>(src1) +
										               (shape1[0] == 1
										                    ? 0
										                    : i * shape1.SubShape(1).NumElements() * sizeof(T1))
										         : static_cast<const std::byte*>(src1),
										     shape2.NumDim() > 1 ? shape2.SubShape(1) : shape2,
										     shape2.NumDim() > 1
										         ? static_cast<const std::byte*>(src2) +
										               (shape2[0] == 1
										                    ? 0
										                    : i * shape2.SubShape(1).NumElements() * sizeof(T2))
										         : static_cast<const std::byte*>(src2),
										     op);
									}
								}
							};
							// 需要根据 OpTrait::ResultShape(shape1, shape2) 分配结果内存
							// 这里假设 dst 已经分配好足够的内存
							switch (OpValue)
							{
							case BinaryOp::Add: {
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return a + b; });
								break;
							}
							case BinaryOp::Subtract: {
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return a - b; });
								break;
							}
							case BinaryOp::Multiply: {
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return a * b; });
								break;
							}
							case BinaryOp::Divide: {
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return a / b; });
								break;
							}
							case BinaryOp::MatMul: {
								const auto resultShapeOrError = OpTrait::ResultShape(shape1, shape2);
								if (!resultShapeOrError)
								{
									throw std::runtime_error(resultShapeOrError.error());
								}
								assert(shape1.NumDim() == 2 && shape2.NumDim() == 2);
								assert(shape1[1] == shape2[0]);
								const auto& resultShape = ShapeView{ *resultShapeOrError };

								for (auto i = 0uz; i < shape1[0]; ++i)
								{
									for (auto j = 0uz; j < shape2[1]; ++j)
									{
										ResultType sum = 0;
										for (auto k = 0uz; k < shape1[1]; ++k)
										{
											sum += static_cast<T1>(static_cast<const T1*>(src1)[i * shape1[1] + k]) *
											       static_cast<T2>(static_cast<const T2*>(src2)[k * shape2[1] + j]);
										}
										static_cast<ResultType*>(dst)[i * shape2[1] + j] = sum;
									}
								}
								break;
							}
							case BinaryOp::Pow:
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return std::pow(a, b); });
								break;
							case BinaryOp::Max:
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return std::max(a, b); });
								break;
							case BinaryOp::Min:
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return std::min(a, b); });
								break;
							case BinaryOp::Less:
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return a < b; });
								break;
							case BinaryOp::Greater:
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return a > b; });
								break;
							case BinaryOp::Equal:
								broadcastOp(resultShape, dst, shape1, src1, shape2, src2,
								            [](const auto& a, const auto& b) { return a == b; });
								break;
							}
						}
					});
				});
			});
		}
		static constexpr void DoReduceOp(CPU& device, ReduceOp reduceOp, void* dst, DataType type, ShapeView shape,
		                                 const void* src, std::size_t axis)
		{
			assert(axis < shape.NumDim());
			EnumDispatch(type, [&]<DataType TypeValue> {
				using T = DataTypeMapping<TypeValue>;
				EnumDispatch(reduceOp, [&]<ReduceOp OpValue> {
					using OpTrait = ReduceOpTraits<OpValue>;
					constexpr auto resultTypeOrError = OpTrait::ResultType(TypeValue);
					if constexpr (!resultTypeOrError)
					{
						throw std::runtime_error(resultTypeOrError.error());
					}
					else
					{
						using ResultType = DataTypeMapping<resultTypeOrError.value()>;

						// 将 shape 分为 outer * reduceDim * inner
						auto outer = 1uz;
						for (auto i = 0uz; i < axis; ++i)
						{
							outer *= shape[i];
						}
						const auto reduceDim = shape[axis];
						auto inner = 1uz;
						for (auto i = axis + 1; i < shape.NumDim(); ++i)
						{
							inner *= shape[i];
						}

						const auto* srcPtr = static_cast<const T*>(src);
						auto* dstPtr = static_cast<ResultType*>(dst);

						for (auto o = 0uz; o < outer; ++o)
						{
							for (auto j = 0uz; j < inner; ++j)
							{
								// 初始化归约值
								auto acc = srcPtr[(o * reduceDim + 0) * inner + j];
								for (auto r = 1uz; r < reduceDim; ++r)
								{
									const auto val = srcPtr[(o * reduceDim + r) * inner + j];
									if constexpr (OpValue == ReduceOp::Sum || OpValue == ReduceOp::Mean)
									{
										acc += val;
									}
									else if constexpr (OpValue == ReduceOp::Max)
									{
										if (val > acc)
										{
											acc = val;
										}
									}
								}
								if constexpr (OpValue == ReduceOp::Mean)
								{
									dstPtr[o * inner + j] =
									    static_cast<ResultType>(acc) / static_cast<ResultType>(reduceDim);
								}
								else
								{
									dstPtr[o * inner + j] = static_cast<ResultType>(acc);
								}
							}
						}
					}
				});
			});
		}

		static constexpr void DoConcatOp(CPU& device, void* dst, DataType type, const void* const* srcPtrs,
		                                  const ShapeView* srcShapes, std::size_t inputCount, std::size_t axis)
		{
			EnumDispatch(type, [&]<DataType TypeValue> {
				using T = DataTypeMapping<TypeValue>;
				// 所有输入除 axis 维度外 shape 相同，计算 outer 和 inner
				const auto& shape0 = srcShapes[0];
				auto outer = 1uz;
				for (auto d = 0uz; d < axis; ++d)
				{
					outer *= shape0[d];
				}
				auto inner = 1uz;
				for (auto d = axis + 1; d < shape0.NumDim(); ++d)
				{
					inner *= shape0[d];
				}

				auto* dstPtr = static_cast<T*>(dst);
				for (auto o = 0uz; o < outer; ++o)
				{
					for (auto i = 0uz; i < inputCount; ++i)
					{
						const auto axisDim = srcShapes[i][axis];
						const auto chunkSize = axisDim * inner;
						const auto* srcPtr = static_cast<const T*>(srcPtrs[i]) + o * axisDim * inner;
						std::copy_n(srcPtr, chunkSize, dstPtr);
						dstPtr += chunkSize;
					}
				}
			});
		}

		static constexpr void DoSliceOp(CPU& device, void* dst, DataType type, ShapeView srcShape, const void* src,
		                                 std::size_t axis, std::size_t start, std::size_t length)
		{
			assert(axis < srcShape.NumDim());
			assert(start + length <= srcShape[axis]);
			EnumDispatch(type, [&]<DataType TypeValue> {
				using T = DataTypeMapping<TypeValue>;
				auto outer = 1uz;
				for (auto d = 0uz; d < axis; ++d)
				{
					outer *= srcShape[d];
				}
				const auto axisDim = srcShape[axis];
				auto inner = 1uz;
				for (auto d = axis + 1; d < srcShape.NumDim(); ++d)
				{
					inner *= srcShape[d];
				}

				const auto srcStride = axisDim * inner;
				const auto dstStride = length * inner;
				const auto* srcPtr = static_cast<const T*>(src);
				auto* dstPtr = static_cast<T*>(dst);

				for (auto o = 0uz; o < outer; ++o)
				{
					std::copy_n(srcPtr + o * srcStride + start * inner, dstStride, dstPtr + o * dstStride);
				}
			});
		}
	};

} // namespace LiteNN

#endif
