# 数据目录

此目录有意拆分为仅本地使用的数据区域：

- `raw/`：由 `scripts/download_kaggle_data.py` 下载的 Kaggle 数据。
- `processed/`：生成的 schema 画像和派生分析表。

这两个区域都会被 Git 忽略。请提交数据字典、源代码和可复现脚本，而不是原始 Kaggle 文件。
