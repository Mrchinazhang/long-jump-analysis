# long-jump-analysis

从逐帧坐标还原一次跳远的技术指标：起跳帧、落地帧、最高点、滞空时长，以及上升段与下降段的平均速度。

给视频里的运动员打点（每帧一个 `x, y`）之后，剩下的计算交给这个脚本。

```
起跳点坐标: (25.00, 18.12) (帧 19, 对应坐标: (25.08, 18.34))
落点坐标: (95.00, 18.12) (帧 280, 对应坐标: (94.92, 18.34))
总帧数: 300 (帧率 30 fps, 1像素=0.1米)

=== 滞空阶段速度分析 ===
上升阶段帧范围: 19 到 149
下降阶段帧范围: 150 到 280
滞空时长: 8.73 秒（抛物线）
实际平均上升速度 (x, y): (0.65, 0.87) 米/秒
实际平均下降速度 (x, y): (0.65, -0.87) 米/秒

=== 滞空阶段抛物线方程 ===
y = -0.0027x² + 0.7010x + 18.6429
```

## 处理流程

```mermaid
flowchart LR
    A["逐帧坐标<br/>N × 2"] --> B["KMeans 聚类<br/>起跳 / 滞空 / 落地"]
    B --> C["按 x 排序<br/>确定起跳点与落点"]
    C --> D["线性插值<br/>重采样到 N 帧"]
    D --> E["滞空段<br/>抛物线拟合"]
    E --> F["顶点 = 最高点<br/>切分上升 / 下降"]
    F --> G["相邻帧位移 × 帧率<br/>× 像素米换算 → 速度"]
```

**一、聚类分段。** 用 KMeans 把坐标点聚成 3 簇，再按簇中心的 x 坐标排序，最左是起跳点、最右是落点。这里假设拍摄方向与助跑方向一致。

**二、拟合找顶点。** 把离散的标注点线性插值成固定帧数的时间序列，对滞空段做二次多项式拟合，顶点位置即最高点，据此切出上升段与下降段。

**三、差分求速度。** 速度 = 相邻帧位移 × 帧率 × 像素-米换算系数。位移与速度都是二维向量，所以 x、y 分量分开输出：x 分量是水平位移速度，y 分量是竖直方向的升降速度。

## 安装

需要 Python 3.8 及以上。

```bash
git clone https://github.com/Mrchinazhang/long-jump-analysis.git
cd long-jump-analysis
pip install -r requirements.txt
```

## 快速开始

```bash
# 用内置示例数据跑一遍
python long_jump_analysis.py --demo

# 交互输入：先填跳远成绩（不知道可直接回车），再逐行粘贴坐标，Ctrl+Z 回车（Windows）或 Ctrl+D（Unix）结束
python long_jump_analysis.py

# 直接把示例文件喂进去
python long_jump_analysis.py < examples/sample_coordinates.txt

# 60 fps 的视频重新算
python long_jump_analysis.py --demo --fps 60

# 只出结果不出图（便于写脚本或跑批）
python long_jump_analysis.py --demo --no-show
```

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--fps` | `30.0` | 录制帧率（帧/秒），决定速度与滞空时长的换算 |
| `--pixel-to-meter` | `0.1` | 像素到米的换算系数，即 1 像素代表多少米 |
| `--frames` | `300` | 重采样后的总帧数，**应设为视频实际帧数** |
| `--clusters` | `3` | KMeans 聚类簇数，对应起跳/滞空/落地三段 |
| `--demo` | 关闭 | 跳过手动输入，使用内置示例轨迹 |
| `--no-show` | 关闭 | 不弹出 matplotlib 图表窗口 |

## 输入数据格式

每行一个坐标点，`x` 与 `y` 之间用空格或逗号分隔：

```
8.12          ← 跳远成绩（米），可省略
20 5
30 31.25
40 50
50 61.25
60 65
70 61.25
80 50
90 31.25
100 5
```

坐标单位只要是"像素"且与 `--pixel-to-meter` 匹配即可。点数必须不少于聚类数，滞空段至少 3 帧，否则无法拟合抛物线。

## 输出字段

`analyze_midair_speed()` 返回一个字典：

| 键 | 含义 |
| --- | --- |
| `peak_frame` | 最高点所在帧（绝对帧号） |
| `rising_frames` / `falling_frames` | 上升段与下降段的帧号列表 |
| `avg_rising_speed` / `avg_falling_speed` | 两段的平均速度向量，单位 米/秒 |
| `duration_s` | 滞空时长，秒 |
| `parabola_coefficients` | 拟合得到的 `(a, b, c)` |
| `fit_mode` | 顶点是如何确定的：`抛物线` / `直线退化…` / `开口向上…` |
| `fps` / `pixel_to_meter` | 本次计算实际使用的参数，便于回溯 |

## 作为库调用

核心函数都在模块顶层，可以直接 import，不必走命令行：

```python
import numpy as np
from long_jump_analysis import analyze_coordinates, analyze_midair_speed

coords = np.array([[20, 5], [40, 50], [60, 65], [80, 50], [100, 5]], dtype=float)

takeoff, landing, labels, centers, frame_coords = analyze_coordinates(coords, frames=300)
tf = int(np.argmin(np.linalg.norm(frame_coords - takeoff, axis=1)))
lf = int(np.argmin(np.linalg.norm(frame_coords - landing, axis=1)))

result = analyze_midair_speed(frame_coords, tf, lf, distance=8.12,
                              fps=60.0, pixel_to_meter=0.05)
print(result["duration_s"], result["avg_rising_speed"])
```

`distance` 传入实测跳远成绩时，速度会按 `distance / distance_ref_m` 的比例缩放；`distance_ref_m` 默认 10 米（`DEFAULT_DISTANCE_REF_M`），可在调用时单独指定。传 0 表示不做缩放。

## 项目结构

```
long-jump-analysis/
├── long_jump_analysis.py       # 主程序，也可作为模块导入
├── tests/
│   └── test_analysis.py        # 23 项回归测试
├── examples/
│   └── sample_coordinates.txt  # 可直接重定向输入的示例
├── requirements.txt
├── LICENSE
└── README.md
```

### 各文件来源说明

| 文件 / 目录 | 来源 | 说明 |
| --- | --- | --- |
| `long_jump_analysis.py` | 作者编写（项目主体） | 核心算法都在这里：KMeans 聚类分段、线性插值重采样、抛物线拟合、上升/下降速度换算，同时支持命令行与 `import` 两种用法。 |
| `tests/test_analysis.py` | 作者补充 | 23 项 `unittest` 回归测试，覆盖参数化精度、退化情形与非法入参，不依赖 pytest。 |
| `examples/sample_coordinates.txt` | 作者提供 | 一份可直接 `python long_jump_analysis.py < examples/sample_coordinates.txt` 喂入的内置示例轨迹。 |
| `requirements.txt` | 项目初始化附带 | 声明运行依赖：`numpy`、`scikit-learn`、`matplotlib`。 |
| `LICENSE` | 项目初始化附带 | MIT 许可证，版权归作者 Mrchinazhang（2026）。 |
| `README.md` | 作者编写 | 你正在读的这份文档。 |
| `.gitignore` | 项目初始化附带 | 忽略 `__pycache__/`、虚拟环境、IDE 配置等，避免把本地杂物提交进仓库。 |
| `.gitattributes` | 仓库创建时生成 | GitHub Desktop 建仓时写入，统一换行符与文本/二进制处理，保持跨平台一致性。 |

> 主体逻辑（KMeans 聚类 + 抛物线拟合 + 速度换算）由作者从零实现；依赖声明与仓库配套文件采用社区通用模板，不重复造轮子。

## 测试

只用标准库 `unittest`，不需要额外装 pytest：

```bash
python -m unittest discover -s tests -v
```

覆盖内容包括：核心函数是否真在模块顶层、`--fps` / `--pixel-to-meter` 参数化是否精确生效（翻倍则速度严格翻倍）、插值重采样是否避免零位移平台、共线与开口向上等退化情形、以及各类非法入参是否抛出可读的 `ValueError`。

## 几个设计上的取舍

**重采样用插值，不用索引抽样。** 早期实现用 `coordinates[frame_indices]` 把 N 个点铺到 300 帧，当 N 远小于帧数时同一个坐标会被重复铺满几十帧，相邻帧位移为 0，速度成片算出 0。改成 `np.interp` 线性插值后，位移连续，且当 `len(coordinates) == frames` 时结果与原实现完全一致。

**顶点要处理三种退化情形。** `np.polyfit` 拟合出 `a ≈ 0` 时曲线是直线、没有内部极值；`a > 0` 时开口向上，顶点是最低点而不是最高点。这两种情况都退回"取纵向坐标最大的那一帧"，并在 `fit_mode` 里标注原因，而不是拿一个数学上无意义的顶点当结果。顶点算出落在区间外时也会裁剪并标注。

**分段用切片而不是帧号换算。** 上升段取 `velocities[:peak_idx]`、下降段取 `velocities[peak_idx:]`，避免把绝对帧号换算成相对下标时越界——这正是早期版本在示例数据上抛 `IndexError` 的原因。

**默认参数全部提到模块常量。** `DEFAULT_FPS`、`DEFAULT_PIXEL_TO_METER`、`DEFAULT_FRAMES` 既是命令行默认值，也能被调用方覆盖，不同设备的参数不必改源码。

## 已知限制

- **贴地坐标会被合并。** KMeans 按空间距离聚类，助跑段与落地后的坐标几乎共线，容易被合并成同一簇，导致起跳点/落点中心偏移。建议输入时只保留从起跳到落地的轨迹段（内置示例就只给了这一段）。
- **单一像素-米系数假设无透视畸变。** 侧面机位、镜头与运动平面平行时误差最小；斜拍需要先做透视校正。
- **只有二维坐标。** 无法还原深度方向的信息，也无法计算真正的三维腾空高度。
- **`--frames` 要与视频实际帧数一致。** 它是"把轨迹重采样成多少帧"，不是越多越好。填得远大于真实帧数，滞空时长会被拉长。
- **图表中文标签需要中文字体。** 代码里优先用 `Microsoft YaHei`、`SimHei`，Linux/macOS 上若未安装会显示成方块，可自行修改 `plot_results()` 中的 `font.sans-serif`。

## 许可

[MIT](LICENSE) © 2026 Mrchinazhang
