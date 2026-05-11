# LiteNN 设计约束

## 必须保持的约束

- Graph 是设备无关的前端 IR，不在 Graph 层注入后端执行细节。
- 高级操作优先通过子图组合表达，不轻易膨胀原语枚举。
- Pass 变换必须在语义上保持 Graph 等价，除非文档明确声明为 lowering / deployment-only extraction。
- `Validation::ValidateGraph` 是公共入口前置防线；新增执行/编译路径时不应绕过它。
- 训练图与推理图的分层保持清晰：forward-only 提取后不再依赖 activation/tape 语义。
- `CompiledModule::Load` 必须保持 image 字节复制语义，调用方传入的原始地址可在 `Load` 返回后释放。

## 当前非目标

- 在 `1.0` 之前承诺完整的 long-term ABI 稳定性。
- 支持任意编译器；当前工具链仍以 reflection-capable GCC/Clang 为主。
- 把 dump 文本、异常字符串或测试命名当作稳定公共协议。

## 修改前需要先判断的事项

- 是不是在扩大 public surface，而不是修内部实现？
- 会不会影响 `LiteNN.h` 和 `import LiteNN;` 两条入口的一致性？
- 会不会改变 model serialization 或 compiled image 的二进制含义？
- 会不会破坏现有 `find_package` / Conan 消费路径？