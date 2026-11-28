# ModelArts 正式训练步骤（对应冻结协议 §7.7；本次正式运行记录）

> 前提：本机已完成 QASPER-only 60,918 条训练样本（答案槽 gold/gen/none 三变体齐）；
> 脚本已验证可跑（CPU 冒烟通过）。正式训练必须在华为云 GPU，不在本机长跑。

## 1. 同步脚本与数据到云端

```bash
# 本机：提交并推送（仓库名/远端按小组实际仓库替换）
cd /home/iah_public/Yusanne/深度学习/大作业/project
git add scripts/ configs/ README.md MODELARTS_STEPS.md
git commit -m "formal training pipeline: reranker + answer-slot protocol"
git push

# 三个数据文件不走 git（体积大），上传 OBS：
#   results/raw/qasper_train_pairs.jsonl      (~51MB)
#   results/raw/gen_answers_train.jsonl       (~1MB)
#   results/raw/gen_manifest_train.jsonl      (~2MB)
# 用 OBS 控制台或 obsutil 上传到 bucket, 例如 obs://citefaith/data/
```

## 2. ModelArts 上执行（ Notebook 或训练作业二选一）

```bash
# Notebook 里（推荐，hw8 同款流程）
git clone <小组仓库> && cd project/scripts
# 从 OBS 挂载/下载三个 jsonl 到 ./data/
python train_modelarts_entry.py \
  --pairs data/qasper_train_pairs.jsonl \
  --gen data/gen_answers_train.jsonl \
  --manifest data/gen_manifest_train.jsonl \
  --epochs 2 --max-length 512 --batch-size 32 --source-filter qasper
```

依赖：`pip install torch transformers rank_bm25 numpy`（GPU 环境自带 torch）。

## 3. 产物回传

训练结束把 `/home/ma-user/work/deeplearning-bighw/formal_qasper_only/` 里的
`epoch*.pt`、`train_log.json`、`last.pt` 下载回本机；checkpoint 不提交到 Git，
只提交训练日志和评测 JSON，并保留 ModelArts 作业日志截图
（可复现性评分项需要训练日志与 checkpoint 对应）。

## 4. 回本机评测

checkpoint 回来后运行 `scripts/s17_eval_reranker.py`，分别评测
`data_splits/eval_main80.json + data/pages` 和
`data_splits/eval_qasper40.json + data/qasper_pages`，同时报告 `none` 与 `gen`
答案槽。门槛：先与 BM25/TF-IDF/RRF/零样本统一比较，再用 paired bootstrap CI
判断增益是否稳定；不能只凭单个 Hit@1 点估计下结论。

## 注意

- 训练数据只来自 train split；运行时评测一律用生成/无答案槽（协议 §3.6.2）；
- 若 GPU 上冒烟报错：先跑 `--epochs 1` + 只取前 500 行数据的小样本验证脚本，
  不要直接烧正式配额；
- 昇腾 NPU 环境如 torch 不可用，改申请 GPU 规格（hw8 经验：昇腾需 torch_npu 适配）。

## 5. 本次实际运行结果（2026-11-28）

- QASPER-only：`formal_qasper_only/epoch2.pt`，训练 loss 0.824→0.546；
- 混合训练：`formal_mixed/epoch2.pt`，训练 loss 0.807→0.538；
- 混合模型 MMLongBench test：生成答案槽 Hit@1=0.583、MRR=0.655；
- 混合模型 QASPER test：生成答案槽 Hit@1=0.500、MRR=0.599；
- 评测 JSON 和训练日志已提交到仓库 `project/results/raw/formal_*`；checkpoint 保留在 ModelArts 工作盘，未提交 Git。

后续不再盲目增加训练轮数；Dense baseline 和 paired bootstrap CI 已完成，下一步优先补 A2/A3/A6 消融与主表。
