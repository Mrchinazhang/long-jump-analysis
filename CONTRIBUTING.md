# 参与贡献

欢迎提 Issue 和 PR，尤其是退化情形的测试用例、不同机位下的误差讨论。

## 本地跑起来

```bash
git clone https://github.com/Mrchinazhang/long-jump-analysis.git
cd long-jump-analysis
pip install -r requirements.txt

python -m unittest discover -s tests -v
python long_jump_analysis.py --demo --no-show
```

## 提交前自查

- 测试只用标准库 `unittest`，**不要引入 pytest**；CI 跑的是 `unittest discover`
- 新增算法分支必须同时补上退化情形测试（共线、开口向上、顶点落在区间外、非法入参）
- 保持参数化：不要把 `fps` / `pixel_to_meter` / `frames` / `clusters` 写死回源码，统一走 `DEFAULT_*` 常量 + 命令行参数
- `requirements.txt` 的大版本上界不要随意放宽：升级可能改变 KMeans 与绘图行为
- 不要提交生成的图表 PNG、`__pycache__/`、虚拟环境或 IDE 配置

## PR 约定

- 一个 PR 一件事，说明「改了什么 + 测试怎么跑的」
- 改动数值结果时，在 PR 里贴改动前后的示例输出，方便核对
- CI 覆盖 Python 3.10 / 3.11 / 3.12 / 3.13，本地至少跑通一个版本（3.9 已 EOL，托管 runner 不再提供）
