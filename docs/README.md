# LiteNN 文档索引

- [Architecture.md](Architecture.md): 核心架构、Roadmap、Pass/Runtime/AOT 设计概览。
- [Roadmap.md](Roadmap.md): 路线图目标树，勾选项是当前进度的依据。
- [ProductionRefactorPlan.md](ProductionRefactorPlan.md): 生产化重构的阶段计划。
- [GGMLCompatibility.md](GGMLCompatibility.md): GGUF / llama.cpp lowering 中哪些 surface 是通用 LiteNN 语义，哪些是兼容性专用约定。
- [PerformanceAnalysis.md](PerformanceAnalysis.md): CPU AOT MatMul 路径的瓶颈分析与优化记录，基线数据来自 2026-05。
- [PerformanceBenchmark_2026-06-18.md](PerformanceBenchmark_2026-06-18.md): 当前标准推理矩阵的 CPU/CUDA/Vulkan/PyTorch/ggml 横向性能快照。
- [PerformanceOptimizationRoadmap.md](PerformanceOptimizationRoadmap.md): 基于 benchmark 和性能分析拆出的优化执行路线图。
- [Versioning.md](Versioning.md): 版本治理、兼容策略、弃用规则、序列化与 image 迁移约定。
- [APIGuide.md](APIGuide.md): 常用 public API 入口、推荐调用路径和使用边界。
- [DesignConstraints.md](DesignConstraints.md): 当前必须保持的设计约束与非目标。
- [PassGuide.md](PassGuide.md): 新增/修改 Graph Pass 的约定、检查点与测试要求。
- [DeviceBackendGuide.md](DeviceBackendGuide.md): 新设备后端的接入要求与验证路径。
- [CUDAAOTRoadmap.md](CUDAAOTRoadmap.md): CUDA AOT native 后端的目标分解与阶段路线图。
- [AOTDeployment.md](AOTDeployment.md): AOT artifact、carrier object、部署和加载建议。
- [VulkanMobileDeployment.md](VulkanMobileDeployment.md): Vulkan-native 移动端打包、loader、validation layer 与 artifact 约束。
- [Troubleshooting.md](Troubleshooting.md): 常见构建、打包、模块、sanitizer 与运行时问题排查。
- [../CHANGELOG.md](../CHANGELOG.md): 用户可见改动记录。

## 历史性能分析

- [PerformanceAnalysis_2026-05-16.md](PerformanceAnalysis_2026-05-16.md)
- [PerformanceAnalysis_2026-05-19.md](PerformanceAnalysis_2026-05-19.md)
- [PerformanceAnalysis_2026-07-02.md](PerformanceAnalysis_2026-07-02.md)
- [PerformanceAnalysis_2026-07-05.md](PerformanceAnalysis_2026-07-05.md)
- [PerformanceAnalysis_2026-07-07.md](PerformanceAnalysis_2026-07-07.md)
- [PerformanceAnalysis_2026-08-04.md](PerformanceAnalysis_2026-08-04.md)

## Qwen 性能证据

- [QwenCPUDecodeBuildControl_2026-08-04.md](QwenCPUDecodeBuildControl_2026-08-04.md)
- [QwenCPUDecodeCrossRuntimeDecision_2026-08-04.md](QwenCPUDecodeCrossRuntimeDecision_2026-08-04.md)
- [QwenCPUDecodeFFNActivationDownDecision_2026-08-04.md](QwenCPUDecodeFFNActivationDownDecision_2026-08-04.md)
- [QwenCPUDecodePairedControl_2026-08-04.md](QwenCPUDecodePairedControl_2026-08-04.md)
- [QwenCPUDecodePerformanceEvidence_2026-08-04.md](QwenCPUDecodePerformanceEvidence_2026-08-04.md)
- [QwenCPUDecodeProjectionProfile_2026-08-04.md](QwenCPUDecodeProjectionProfile_2026-08-04.md)
- [QwenCPUDecodeStageControl_2026-08-04.md](QwenCPUDecodeStageControl_2026-08-04.md)
- [QwenCPUDecodeActivationDownSplit_2026-08-09.md](QwenCPUDecodeActivationDownSplit_2026-08-09.md)
- [QwenCPUDecodeGNUStageControl_2026-08-09.md](QwenCPUDecodeGNUStageControl_2026-08-09.md)
- [QwenCPUDecodePositionStageControl_2026-08-09.md](QwenCPUDecodePositionStageControl_2026-08-09.md)
- [QwenCPUDecodeStageComparison_2026-08-09.md](QwenCPUDecodeStageComparison_2026-08-09.md)
- [QwenCPUDecodeSustained128Control_2026-08-09.md](QwenCPUDecodeSustained128Control_2026-08-09.md)
- [QwenCPUDecodeSwiGLUEvidence_2026-08-09.md](QwenCPUDecodeSwiGLUEvidence_2026-08-09.md)
- [QwenCPUDecodeAttentionThreadPoolEvidence_2026-08-12.md](QwenCPUDecodeAttentionThreadPoolEvidence_2026-08-12.md)
- [QwenFirstCacheMemoryEvidence_2026-08-12.md](QwenFirstCacheMemoryEvidence_2026-08-12.md)
- [QwenPostNeoXThroughputControl_2026-08-14.md](QwenPostNeoXThroughputControl_2026-08-14.md)
- [QwenEqualThreadScalingControl_2026-08-18.md](QwenEqualThreadScalingControl_2026-08-18.md)
- [QwenEqualThreadScalingCorrectedFormal_2026-08-18.md](QwenEqualThreadScalingCorrectedFormal_2026-08-18.md)
- [QwenInProcessDecodeControl_2026-08-18.md](QwenInProcessDecodeControl_2026-08-18.md)
- [QwenCPUDecodeBoundedActivationEvidence_2026-08-19.md](QwenCPUDecodeBoundedActivationEvidence_2026-08-19.md)
- [QwenT1StageWindowAudit_2026-09-22.md](QwenT1StageWindowAudit_2026-09-22.md)

## Qwen 正确性证据

- [QwenDefaultThreadDeadlockAnalysis_2026-08-12.md](QwenDefaultThreadDeadlockAnalysis_2026-08-12.md)
- [QwenFinalLogitBoundaryAnalysis_2026-08-12.md](QwenFinalLogitBoundaryAnalysis_2026-08-12.md)
- [QwenFirstDivergenceRoPEAnalysis_2026-08-12.md](QwenFirstDivergenceRoPEAnalysis_2026-08-12.md)
- [QwenNaturalDecodeLayerDriftEvidence_2026-08-12.md](QwenNaturalDecodeLayerDriftEvidence_2026-08-12.md)
- [QwenNaturalGenerationQualityAnalysis_2026-08-12.md](QwenNaturalGenerationQualityAnalysis_2026-08-12.md)
- [QwenSubLayerDriftAnalysis_2026-08-12.md](QwenSubLayerDriftAnalysis_2026-08-12.md)
- [QwenNeoXRoPEFixEvidence_2026-08-13.md](QwenNeoXRoPEFixEvidence_2026-08-13.md)
- [QwenCorpusLossAnalysis_2026-08-14.md](QwenCorpusLossAnalysis_2026-08-14.md)
- [QwenFixedReferenceTrajectoryAnalysis_2026-08-14.md](QwenFixedReferenceTrajectoryAnalysis_2026-08-14.md)
