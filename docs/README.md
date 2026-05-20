# LiteNN 文档索引

- [Architecture.md](Architecture.md): 核心架构、Roadmap、Pass/Runtime/AOT 设计概览。
- [GGMLCompatibility.md](GGMLCompatibility.md): GGUF / llama.cpp lowering 中哪些 surface 是通用 LiteNN 语义，哪些是兼容性专用约定。
- [PerformanceAnalysis.md](PerformanceAnalysis.md): 当前 CPU AOT 性能路径与瓶颈分析。
- [PerformanceOptimizationRoadmap.md](PerformanceOptimizationRoadmap.md): 基于 benchmark 和性能分析拆出的优化执行路线图。
- [Versioning.md](Versioning.md): 版本治理、兼容策略、弃用规则、序列化与 image 迁移约定。
- [APIGuide.md](APIGuide.md): 常用 public API 入口、推荐调用路径和使用边界。
- [DesignConstraints.md](DesignConstraints.md): 当前必须保持的设计约束与非目标。
- [PassGuide.md](PassGuide.md): 新增/修改 Graph Pass 的约定、检查点与测试要求。
- [DeviceBackendGuide.md](DeviceBackendGuide.md): 新设备后端的接入要求与验证路径。
- [CUDAAOTRoadmap.md](CUDAAOTRoadmap.md): CUDA AOT native 后端的目标分解与阶段路线图。
- [AOTDeployment.md](AOTDeployment.md): AOT artifact、carrier object、部署和加载建议。
- [Troubleshooting.md](Troubleshooting.md): 常见构建、打包、模块、sanitizer 与运行时问题排查。
- [../CHANGELOG.md](../CHANGELOG.md): 用户可见改动记录。
