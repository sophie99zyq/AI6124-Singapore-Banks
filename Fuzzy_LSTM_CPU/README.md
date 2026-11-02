# Fuzzy LSTM CPU 快速运行

打开 `Fuzzy_LSTM_CPU_Quickstart.ipynb`，默认运行普通 LSTM 和 FuzzyLSTM，各 5 轮。

## VS Code

安装 Microsoft Python 和 Jupyter 扩展。终端进入本目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在 Notebook 右上角选择 `.venv/bin/python`，然后 Run All。已经通过终端安装依赖时，可以跳过 Notebook 安装单元。若在 Notebook 安装，安装后重启 Kernel，再从环境检查单元开始运行。

Python 3.12 / Apple Silicon macOS 为此版本的目标环境。首次下载真实 Yahoo Finance 行情需要联网；未包含模拟数据或预先训练的结果。

## 与原始代码的关系

模型和数据处理函数来自 `../Paper_Fuzzy_LSTM-main/Tests_Fuzzy.ipynb`。保留模型结构、模糊公式、原始 train/test 处理差异及指标逻辑。兼容导入改为 tf_keras，样式改为本地默认；epochs 从 100 改为 5，进度显示开启，参数集中配置；增加共享数据缓存、数据检查和结果汇总，vanilla_LSTM 额外返回模型。

默认分钟配置的 units=16、lookback=2、window=10 与最新原始 Notebook 一致。恢复 epochs=100 可以重跑该版本；论文分钟配置 lookback=30，但论文与最新代码版本存在结构差异，不能仅通过参数保证复现论文数字。

当前仍保留原代码中基线归一化泄漏、测试集兼作验证集、rolling std 起始 bfill 等限制，以便对照源代码。

指标表格使用 pandas 输出，避免 PrettyTable/wcwidth 版本兼容问题；指标数值与模型算法不变。
