# deeplearning-bighw

深度学习课程大作业：面向长文档问答的冗余感知证据忠实性评估与重排序。

## 项目内容

- MMLongBench-Doc：页级长文档主实验；
- QASPER：段落级外部验证和证据监督；
- Cross-Encoder Evidence Reranker：可训练的深度学习模块；
- CiteFaith-20：少量人工删除/替换证据审计；
- Evidence Court：离线答辩展示。

## 仓库内容

仓库保留代码、配置、实验协议、阅读笔记、分工文件、小型统计结果和可复现说明。原始 PDF、QASPER 原始包、逐页文本、模型权重和大规模 JSONL 训练流不上传，由 `project/README.md` 中的脚本重新生成。

主要入口：

- [选题策划和可行性分析.md](选题策划和可行性分析.md)：数据集、基线、指标、阅读笔记、分工和周计划；
- [project/README.md](project/README.md)：实验脚本与运行顺序；
- [阅读笔记.md](阅读笔记.md)：独立版文献笔记；
- [对接说明.md](对接说明.md)：标注规则和线上协作说明；
- `project/configs/`：冻结协议和训练配置；
- `project/scripts/`：数据处理、评测和训练脚本；
- `project/results/raw/review_task_*.xlsx`：独立证据审计任务表。

## 当前状态

数据下载、证据对齐、零样本重排序、QASPER 训练对构造和答案生成已完成；QASPER-only 正式 GPU 训练、Dense baseline 和最终主表仍待运行。当前已有结果不得被表述为正式微调增益。

## 协作约定

1. 每次提交只包含一个清晰模块；
2. 训练配置、随机种子和结果文件与代码一起提交；
3. 不提交 API key、原始数据、模型权重和个人填写副本；
4. 大文件通过脚本下载或在本地/云端生成，不直接进入 Git 历史。
