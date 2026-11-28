# CiteFaith 项目工程目录

题目：有引用，不等于有依赖——面向长文档问答的冗余感知证据忠实性评估与重排序
策划文档：`../选题策划和可行性分析.md`（含冻结协议 §7 与冲刺执行清单）

## 华为云硬件兼容性提醒

课程 homework7/8 提供的 ModelArts Notebook 是昇腾 NPU + MindSpore 环境；本项目的 `s16_train_formal.py` 和 `s17_eval_reranker.py` 当前基于 PyTorch Transformers，命令中的 `--device cuda` 只适用于 CUDA GPU。不要在课程 NPU 上直接运行该命令。

创建实例后先核验：

```bash
python -c "import mindspore as ms; print(ms.__version__)"
npu-smi info
python -c "import torch; print(torch.__version__)"
python -c "import torch_npu; print('torch_npu ok')"
```

如果只有 NPU 而没有可用的 `torch_npu`，需要先把训练脚本适配到 `torch_npu`/Ascend，或向老师申请 CUDA GPU 资源；不要自行混装不匹配的 PyTorch、CANN 和 `torch_npu` 版本。

## 运行顺序（venv python -I，脚本与数据分离）

```bash
PY=/home/iah_public/Yusanne/深度学习/.venv/bin/python
W1=../week1/data            # samples.json / doc_list.json（week1 已下载的 16 篇 PDF 复用其 documents/）

# 1 冻结划分（秒级）
$PY -I scripts/s1_freeze_splits.py $W1/samples.json data_splits $W1/selected20.json

# 2 下载全库 135 篇 PDF（week1 的 16 篇复用，其余由官方 GitHub 文档清单补齐）
$PY -I scripts/s2_fetch_docs.py $W1/doc_list.json data/documents

# 3 逐页文本抽取
$PY -I scripts/s3_extract_all.py data/documents data/pages

# 4 审计层自动筛选（验证 ≥20 门槛 + 产出复核清单）
$PY -I scripts/s4_autofilter_audit.py data_splits/audit_pool.json data/pages results/raw

# 5 MMLongBench-Doc 训练对构造（文档隔离；当前实际 pilot 为 15 题/660 对）
$PY -I scripts/s5_build_train_pairs.py $W1/samples.json data_splits \
      results/raw/audit_candidates.json data/pages results/raw

# 6 生成答案槽（qwen-plus，train split 专用，缓存续跑）
OPENAI_API_KEY=... OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
$PY -I scripts/s6_gen_train_answers.py results/raw/gen_manifest.jsonl results/raw/gen_answers.jsonl

# 7 零样本 reranker 基线行（本地 CPU；需 transformers；模型已冻结为 MiniLM-L6）
$PY -I scripts/s7_zeroshot_rerank.py data_splits/eval_main80.json data/pages results/raw

# 8 QASPER 官方证据版本：构造段落单位的 40 题外部验证切片
$PY -I scripts/s11_prepare_qasper.py data/qasper_official/raw_train_dev/qasper-train-v0.3.json \
      data/qasper_official/raw_train_dev/qasper-dev-v0.3.json \
      data/qasper_official/raw_test/qasper-test-v0.3.json \
      data/qasper_pages data_splits/eval_qasper40.json results/raw/qasper_stats.json

# 9 在 QASPER 外部切片上跑同一套检索/重排序评测
$PY -I scripts/s7_zeroshot_rerank.py data_splits/eval_qasper40.json data/qasper_pages results/raw/qasper_zeroshot
# 该切片现在 strict_evidence=true，只使用官方 evidence 段落，不扩展邻段。
$PY -I scripts/s13_bootstrap_ci.py results/raw/qasper_zeroshot/zeroshot_rerank.json \
      results/raw/qasper_zeroshot/bootstrap_ci.json 2000

# 10 构造 QASPER train 的段落 evidence 对（已实测 2,097 题 / 40,630 对）
$PY -I scripts/s12_build_qasper_pairs.py data/qasper_official/raw_train_dev/qasper-train-v0.3.json \
      results/raw/qasper_train_pairs.jsonl results/raw/qasper_train_pairs_stats.json

# 11 QASPER train evidence 对的正式训练（ModelArts；本机只跑 smoke test）
# qasper_train_pairs.jsonl 保留 40,630 行官方 gold/none 对；
# s16 会按 question-document-page-label 去重，再展开可用的 gold/gen/none 槽，
# 当前预检得到 60,918 个实际训练样本。
$PY -I scripts/s16_train_formal.py \
  --pairs results/raw/qasper_train_pairs.jsonl \
  --gen results/raw/gen_answers_train.jsonl \
  --manifest results/raw/gen_manifest_train.jsonl \
  --out models/qasper_supervised_v1 --source-filter qasper \
  --epochs 2 --max-length 512 --batch-size 32 --device cuda

# 11b 训练后分别评测两个数据源（指标不能合并）
$PY -I scripts/s17_eval_reranker.py models/qasper_supervised_v1 \
  data_splits/eval_main80.json data/pages results/raw/qasper_model_mm_eval \
  --answers results/raw/gen_answers_eval.jsonl --device cuda
$PY -I scripts/s17_eval_reranker.py models/qasper_supervised_v1 \
  data_splits/eval_qasper40.json data/qasper_pages results/raw/qasper_model_qasper_eval \
  --answers results/raw/gen_answers_eval.jsonl --device cuda

# 12 混合训练（已于 2026-11-28 在 ModelArts 完成；原始文件 40,630 QASPER + 660 MMLongBench；
# s16 展开为 61,908 个实际训练样本）
$PY -I scripts/s14_merge_pairs.py results/raw/mixed_train_pairs.jsonl \
      qasper=results/raw/qasper_train_pairs.jsonl \
      mmlongbench_doc=results/raw/train_pairs.jsonl
# 配置：configs/modelarts_mixed.yaml；不能把两个数据源的指标合并。
$PY -I scripts/s16_train_formal.py \
  --pairs results/raw/mixed_train_pairs.jsonl \
  --gen results/raw/gen_answers_train.jsonl \
  --manifest results/raw/gen_manifest_train.jsonl \
  --out models/qasper_mmlongbench_mixed_v1 \
  --epochs 2 --max-length 512 --batch-size 32 --device cuda

# 已归档结果：results/raw/formal_mixed_mm_eval/reranker_eval.json
# MMLongBench test finetuned_gen: Hit@1=0.583, MRR=0.655
# QASPER test finetuned_gen: Hit@1=0.500, MRR=0.599
# 两个数据源必须分开报告，不能合并 Hit@k。

# 13 Dense baseline 与 CI（本机推理，不需要 GPU）
$PY -I scripts/s18_dense_rerank.py data_splits/eval_main80.json data/pages \
  results/raw/dense_mm_eval --device cpu
$PY -I scripts/s18_dense_rerank.py data_splits/eval_qasper40.json data/qasper_pages \
  results/raw/dense_qasper_eval --device cpu
$PY -I scripts/s19_bootstrap_ci.py \
  results/raw/formal_mixed_mm_eval/reranker_eval.json \
  results/raw/formal_mixed_mm_eval/bootstrap_ci.json 2000 rrf
$PY -I scripts/s19_bootstrap_ci.py \
  results/raw/formal_mixed_qasper_eval/reranker_eval.json \
  results/raw/formal_mixed_qasper_eval/bootstrap_ci.json 2000 rrf
$PY -I scripts/s20_compare_eval_ci.py \
  results/raw/formal_qasper_mm_eval/reranker_eval.json \
  results/raw/formal_mixed_mm_eval/reranker_eval.json \
  results/raw/formal_mixed_vs_qasper_mm_ci.json 2000 finetuned_gen

# 13 表格转换（评测 CSV 全部由此生成 LaTeX，禁止手敲数字）
$PY -I scripts/s9_csv_to_latex.py results/tables/main.csv results/tables/main.tex --bold-max
```

## 目录

- `scripts/` 全部实验脚本（自包含，`lib_cf.py` 为共享库）
- `configs/frozen_protocol.yaml` 冻结常数
- `data_splits/` 划分与审计清单（版本控制）
- `data/` 原始文档与逐页文本（不入库，README 记录取得方式）
- `results/raw|tables/` 原始输出与论文表格
- `models/` reranker checkpoint 与训练日志（ModelArts 侧产出后回传）

## 证据干预案例文件（可选）

主实验不要求完成 20 题人工标注，也不把人工复核作为训练或主指标的前置条件。下面文件是前期 pilot 的历史任务包；如果时间允许，可从中挑选不超过 5 个案例用于答辩展示和定性说明：

- `results/raw/review_assignments.csv`：历史 20 题分工表；
- `results/raw/review_task_pan.xlsx`：前期 pilot 任务文件，可选抽查时参考；
- `results/raw/review_task_zhang.xlsx`：前期 pilot 任务文件，可选抽查时参考；
- `results/raw/review_task_pan.csv` / `review_task_zhang.csv`：便于程序读取的任务清单。

如果进行可选抽查，只填写对应案例的黄色列；不要求合并 20 题，也不计算大规模一致性。自动证据干预的主结果由程序直接生成。

## 边界

- API key 只走环境变量，绝不写入任何文件；
- 运行时（审计与最终系统）答案槽一律用生成答案，金标只出现在 train split；
- 当前 TF-IDF 是 Dense 的临时代理；正式 Dense baseline 固定使用 `sentence-transformers/all-MiniLM-L6-v2`，结果表必须明确区分代理结果和正式 Dense 结果。
- LongDocURL 属于 MMLongBench 内的任务，且主要是页面图像，不是本项目默认的独立第二数据集；默认外部验证使用官方 QASPER 段落证据。
