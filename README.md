# Subscription Churn Retention Modeling

这是一个用于订阅流失预测、留存分析以及 OTT 观众中途流失风险建模的作品集项目。

本仓库使用两个公开 Kaggle 数据集作为可复现的数据源，并将所有项目代码、分析和报告保留在本仓库本地。

## 数据来源

| 数据集 | Kaggle slug | 用途 |
| --- | --- | --- |
| Customer Subscription Churn and Usage Patterns | `jayjoshi37/customer-subscription-churn-and-usage-patterns` | 客户级流失预测与留存策略分析 |
| OTT Viewer Drop-Off and Retention Risk Dataset | `eklavya16/ott-viewer-drop-off-and-retention-risk-dataset` | 剧集级中途流失预测与内容留存风险分析 |

这些数据集属于教学/合成数据源。本仓库中的结果应理解为作品集与建模演示，而不是关于真实用户群体的结论。

## 项目结构

```text
.
|-- data/
|   |-- raw/                  # Kaggle 下载数据，Git 忽略
|   `-- processed/            # 本地画像分析输出，Git 忽略
|-- notebooks/                # EDA 与建模 notebook
|-- reports/
|   `-- figures/              # 生成的图表，Git 忽略
|-- scripts/                  # 下载、验证、画像分析和训练命令
|-- src/churn_retention/      # 可复用的项目包
`-- tests/                    # 本地项目代码的单元测试
```

## 环境配置

使用 Python 3.11。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m ipykernel install --user --name churn-retention --display-name "Python (churn-retention)"
```

验证环境：

```powershell
.\.venv\Scripts\python scripts\validate_environment.py
```

## Kaggle 配置

从你的 Kaggle 账号设置中下载 `kaggle.json`，并将其放到：

```powershell
%USERPROFILE%\.kaggle\kaggle.json
```

然后下载两个数据集：

```powershell
.\.venv\Scripts\python scripts\download_kaggle_data.py
```

该命令会将文件写入 `data/raw/`，该目录已被有意加入 Git 忽略。

## 运行项目

分析已下载数据：

```powershell
.\.venv\Scripts\python scripts\profile_data.py
```

训练基线模型并写入指标/图表：

```powershell
.\.venv\Scripts\python scripts\train_baselines.py
```

运行测试与 lint：

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
```

## 建模范围

本项目包含两条监督式建模路径：

- 订阅流失：基于订阅套餐、费用、参与度、客服支持、支付、订阅时长和近期活动信号来预测客户流失。
- OTT 中途流失：基于节奏、开场吸引力、内容元数据、观看行为和认知负荷特征来预测剧集级中途流失。

基线模型包括逻辑回归和基于树的模型。评估报告包括 ROC-AUC、PR-AUC、分类报告、类别平衡图，以及在模型支持时提供的特征重要性摘要。

## 备注

- 原始 Kaggle 数据永不提交。
- 公开报告应引用 Kaggle 数据集页面，并说明其合成/教学用途。
- Kaggle notebook 可以作为有用参考，但本仓库的源代码和分析均在本地编写，以支持可复现性和作品集评审。
