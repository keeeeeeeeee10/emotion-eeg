# 从这里开始

这个目录是准备 push 到 GitHub 的干净代码版。

先看：

```text
README.md
```

方法说明页面：

```text
docs/research_method_results.html
```

最终提交文件不会随仓库上传，复现实验后会生成在：

```text
outputs/weighted_offset_source_ensemble_main_w0134/public_test_submission_weighted_offset_source_ensemble_main_w0134_top4.xlsx
```

最终融合脚本：

```text
scripts/run_weighted_offset_source_ensemble.py
```

默认数据目录应放在仓库根目录：

```text
SEED/SEED_EEG/
MODMA/
赛题四数据集及说明文档/
```

这些数据目录和 `outputs/` 已经写入 `.gitignore`，不会被误传。

当前本地 public-like subject-wise proxy 最好结果：

```text
accuracy = 0.8125
HC accuracy = 0.8563
DEP accuracy = 0.7250
```

注意：这个分数来自赛题训练集构造的 public-like proxy，不是公开测试集真实标签分数。
