#include <print>

// import LiteNN;
#include <LiteNN.h>

using namespace LiteNN;

int main()
{
	Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	std::println("tensor: {}", tensor);
	const auto view = tensor[0];
	std::println("view: {}", view);
	const auto addResult = tensor + tensor;
	std::println("addResult: {}", addResult);
	const auto transposeResult = tensor.Transpose();
	std::println("transposeResult: {}", transposeResult);
	const auto matmulResult = tensor.MatMul(transposeResult);
	std::println("matmulResult: {}", matmulResult);

	// 测试广播
	Tensor<CPU> tensor2({ 10, 20 }, { 2, 1 });
	std::println("tensor2: {}", tensor2);
	const auto addResult2 = tensor + tensor2;
	std::println("addResult2: {}", addResult2);

	// 测试设备间复制
	const auto polyTensor = tensor.CopyToDevice(PolymorphicDevice{ CPU{} });
	std::println("polyTensor: {}", polyTensor);
}
