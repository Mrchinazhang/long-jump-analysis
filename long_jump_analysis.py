"""跳远运动的坐标聚类分析与滞空阶段速度估算。

从一段视频里逐帧标注运动员的坐标后，本工具完成三件事：

1. 用 KMeans 把坐标点聚成 起跳 / 滞空 / 落地 三段，按 x 坐标排序确定起跳点与落点；
2. 把离散采样点线性插值成固定帧率的时间序列，对滞空段做抛物线拟合，定位最高点，
   从而切出上升段与下降段；
3. 按「相邻帧位移 × 帧率 × 像素-米换算」算出两段的平均速度，并可用实测跳远成绩
   对速度做比例校正。

用法：
    python long_jump_analysis.py                  # 交互输入坐标与成绩
    python long_jump_analysis.py --demo           # 用内置示例数据跑一遍
    python long_jump_analysis.py --demo --fps 60  # 换成 60fps 重算
    python long_jump_analysis.py --help           # 查看全部参数

作为库使用：
    from long_jump_analysis import analyze_coordinates, analyze_midair_speed
"""
from __future__ import annotations

from typing import Any, Optional

import argparse

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------- 默认参数（均可被命令行覆盖） ----------------
DEFAULT_FRAMES = 300           # 重采样后的总帧数
DEFAULT_FPS = 30.0             # 录制帧率，帧/秒
DEFAULT_PIXEL_TO_METER = 0.1   # 1 像素对应多少米
DEFAULT_DISTANCE_REF_M = 10.0  # 成绩缩放的参考距离，米
DEFAULT_CLUSTERS = 3           # 起跳 / 滞空 / 落地


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """命令行参数。不带参数运行时行为与原来完全一致（走交互输入）。"""
    parser = argparse.ArgumentParser(
        description="运动员坐标聚类分析与滞空阶段速度估算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help="录制帧率（帧/秒）")
    parser.add_argument("--pixel-to-meter", type=float, default=DEFAULT_PIXEL_TO_METER,
                        help="像素到米的换算系数（1 像素 = 多少米）")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="重采样后的总帧数")
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS,
                        help="KMeans 聚类簇数")
    parser.add_argument("--demo", action="store_true",
                        help="跳过手动输入，直接使用内置示例数据")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹出图表窗口（便于自动化运行）")
    return parser.parse_args(argv)


def demo_coordinates() -> np.ndarray:
    """内置示例数据：一条完整的滞空轨迹（起跳点 → 顶点 → 落地点）。

    坐标单位为像素（与视频标注一致）。y 按
    y = 5 + 0.0375 (x - 20)(100 - x) 生成，顶点在 x = 60、y = 65，
    x 方向跨度 80 像素 ≈ 8 米（按 1 像素 = 0.1 米换算）。

    注意这里只给滞空段：若把助跑与落地后的贴地坐标一起传进来，
    KMeans 会把它们合并成同一簇（它们几乎共线），起跳/落点的中心就会偏移。
    数据仅用于验证流程，不代表真实运动员的物理量。
    """
    arc = [[x, 5 + 0.0375 * (x - 20) * (100 - x)] for x in range(20, 101, 10)]
    return np.array(arc, dtype=float)


def input_coordinates(use_demo: bool = False) -> tuple[np.ndarray, float]:
    """手动输入运动员的坐标数据和跳远成绩（起跳前、滞空、落地后）。"""
    if use_demo:
        print("使用内置示例数据（跳过手动输入）")
        return demo_coordinates(), 0.0

    # 输入跳远成绩（可选）
    print("请输入运动员的跳远成绩（米，如果不知道请直接按Enter键跳过）：")
    distance_input = input()
    distance = 0.0
    if distance_input.strip():
        try:
            distance = float(distance_input)
        except ValueError:
            print("成绩输入不是有效数字，已按 0 处理（不做成绩缩放）")

    # 再输入坐标数据
    print("请依次输入运动员的坐标数据（每行为x,y坐标，用空格或逗号分隔）：")
    print("输入完成后按Enter键，然后按Ctrl+Z(Windows)或Ctrl+D(Unix)结束输入")

    coordinates = []
    while True:
        try:
            line = input()
            # 处理分隔符
            line = line.replace(",", " ").split()
            if len(line) != 2:
                print("每行必须包含两个数字（x和y坐标），已忽略此行")
                continue
            x, y = map(float, line)
            coordinates.append([x, y])
        except EOFError:
            break
        except ValueError:
            print("输入包含非数字内容，已自动忽略")
            continue

    if not coordinates:
        print("未输入有效数据，将使用示例数据")
        return demo_coordinates(), distance

    return np.array(coordinates, dtype=float), distance


def analyze_coordinates(coordinates: np.ndarray, frames: int = DEFAULT_FRAMES,
                        clusters: int = DEFAULT_CLUSTERS,
                        random_state: int = 0) -> tuple[np.ndarray, ...]:
    """使用K-means聚类分析坐标数据，并把轨迹重采样到指定帧数。"""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("坐标数据必须是 N×2 的二维数组（每行一个 x,y）")
    if len(coordinates) < clusters:
        raise ValueError(
            f"有效坐标点只有 {len(coordinates)} 个，少于聚类数 {clusters}，无法划分"
            "起跳/滞空/落地三个阶段，请至少输入 3 个点")
    if frames < 2:
        raise ValueError(f"总帧数必须 >= 2，当前为 {frames}")

    kmeans = KMeans(n_clusters=clusters, random_state=random_state, n_init=10).fit(coordinates)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 根据聚类中心的位置判断起跳、滞空、落地
    # 假设起跳点在左侧，落点在右侧
    sorted_indices = np.argsort(centers[:, 0])
    takeoff = centers[sorted_indices[0]]
    landing = centers[sorted_indices[-1]]

    # 把原始采样点线性插值到 frames 帧。
    # 原来用坐标[frame_indices] 做索引抽样：原始点只有几个时，同一坐标会被
    # 重复铺满几十帧，相邻帧位移为 0，算出来的速度成片是 0。
    # 改用插值后，len(coordinates)==frames 时结果与原来完全一致。
    src = np.arange(len(coordinates))
    dst = np.linspace(0, len(coordinates) - 1, frames)
    frame_coords = np.column_stack([
        np.interp(dst, src, coordinates[:, 0]),
        np.interp(dst, src, coordinates[:, 1]),
    ])

    return takeoff, landing, labels, centers, frame_coords


def analyze_midair_speed(frame_coords: np.ndarray, takeoff_frame: int, landing_frame: int,
                         distance: float = 0.0, fps: float = DEFAULT_FPS,
                         pixel_to_meter: float = DEFAULT_PIXEL_TO_METER,
                         distance_ref_m: float = DEFAULT_DISTANCE_REF_M) -> dict[str, Any]:
    """根据实际坐标数据和跳远成绩动态计算滞空阶段的速度。

    参数
    ----
    frame_coords : (N, 2) 数组，每帧一个坐标
    takeoff_frame, landing_frame : 起跳帧与落地帧（含）
    distance : 跳远成绩（米），<=0 表示不做成绩缩放
    fps : 录制帧率，帧/秒
    pixel_to_meter : 像素到米的换算系数
    distance_ref_m : 成绩缩放的参考距离
    """
    if landing_frame <= takeoff_frame:
        raise ValueError(
            f"落地帧({landing_frame})必须大于起跳帧({takeoff_frame})，请检查坐标输入顺序")
    if fps <= 0:
        raise ValueError(f"帧率必须为正数，当前为 {fps}")
    if pixel_to_meter <= 0:
        raise ValueError(f"像素-米换算系数必须为正数，当前为 {pixel_to_meter}")

    # 动态划分上升和下降阶段：只取滞空区间
    midair_coords = frame_coords[takeoff_frame:landing_frame + 1]
    if len(midair_coords) < 3:
        raise ValueError(
            f"滞空阶段只有 {len(midair_coords)} 帧，不足 3 帧，无法拟合抛物线")

    # 使用抛物线拟合y坐标数据
    x = np.arange(len(midair_coords))
    y = midair_coords[:, 1]
    a, b, c = np.polyfit(x, y, 2)

    # 计算最高点（顶点）。
    # 退化情形：a ≈ 0 时是直线，没有内部极值；a > 0 时开口向上，顶点是最低点，
    # 两者都不能当作"最高点"。此时退回"纵向坐标最大的那一帧"。
    if abs(a) < 1e-12:
        peak_idx = int(np.argmax(y))
        fit_mode = "直线退化，无内部最高点(取最高帧)"
    elif a > 0:
        peak_idx = int(np.argmax(y))
        fit_mode = "拟合开口向上，顶点是最低点(取最高帧)"
    else:
        peak_idx = int(round(-b / (2 * a)))
        fit_mode = "抛物线"

    # 顶点必须落在滞空区间内，否则说明拟合与数据不符
    clamped = min(max(peak_idx, 0), len(midair_coords) - 1)
    if clamped != peak_idx:
        fit_mode += "(顶点越界已裁剪)"
        peak_idx = clamped
    peak_frame = takeoff_frame + peak_idx

    # 划分上升和下降阶段（绝对帧号，仅用于输出展示）
    rising_frames = list(range(takeoff_frame, peak_frame + 1))
    falling_frames = list(range(peak_frame + 1, landing_frame + 1))

    # 计算位移与速度：位移单位是像素/帧，乘帧率得像素/秒，再乘换算系数得米/秒
    displacements = np.diff(midair_coords, axis=0)
    velocities = displacements * fps * pixel_to_meter

    # 用段索引切片，避免绝对帧号换算成相对下标时越界（原实现的崩溃点）
    rising_velocities = velocities[:peak_idx]
    falling_velocities = velocities[peak_idx:]

    avg_rising_speed = (np.mean(rising_velocities, axis=0)
                        if len(rising_velocities) else np.zeros(2))
    avg_falling_speed = (np.mean(falling_velocities, axis=0)
                         if len(falling_velocities) else np.zeros(2))

    # 结合跳远成绩调整速度（如果成绩不为0）
    if distance > 0:
        avg_rising_speed = avg_rising_speed * (distance / distance_ref_m)
        avg_falling_speed = avg_falling_speed * (distance / distance_ref_m)

    return {
        "rising_frames": rising_frames,
        "falling_frames": falling_frames,
        "peak_frame": peak_frame,
        "avg_rising_speed": avg_rising_speed,
        "avg_falling_speed": avg_falling_speed,
        "parabola_coefficients": (a, b, c),
        "fit_mode": fit_mode,
        "fps": fps,
        "pixel_to_meter": pixel_to_meter,
        "duration_s": len(midair_coords) / fps,
    }


def plot_results(coordinates: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                takeoff: np.ndarray, landing: np.ndarray) -> None:
    """绘制坐标分布和聚类结果"""
    # 中文字体：不设置时 matplotlib 默认字体没有中文字形，标题会显示成方块
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))

    # 过滤掉(0,0)坐标点
    valid_indices = np.where((coordinates[:, 0] != 0) | (coordinates[:, 1] != 0))[0]
    filtered_coords = coordinates[valid_indices]
    filtered_labels = labels[valid_indices]

    # 绘制原始坐标点
    plt.scatter(filtered_coords[:, 0], filtered_coords[:, 1], c=filtered_labels,
                cmap="viridis", label="坐标点")

    # 绘制聚类中心
    plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="聚类中心")

    # 标记起跳和落点
    plt.scatter(takeoff[0], takeoff[1], c="blue", marker="*", s=300, label="起跳点")
    plt.scatter(landing[0], landing[1], c="green", marker="*", s=300, label="落点")

    plt.title("运动员坐标聚类分析 (已过滤(0,0)坐标)")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.legend()
    plt.grid(True)
    plt.show()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    print("=== 运动员坐标聚类分析 ===")
    coords, distance = input_coordinates(args.demo)

    takeoff, landing, labels, centers, frame_coords = analyze_coordinates(
        coords, frames=args.frames, clusters=args.clusters)

    # 找到起跳和落地对应的帧数
    takeoff_frame = int(np.argmin(np.linalg.norm(frame_coords - takeoff, axis=1)))
    landing_frame = int(np.argmin(np.linalg.norm(frame_coords - landing, axis=1)))

    # 获取起跳和落地对应的具体坐标
    takeoff_coord = frame_coords[takeoff_frame]
    landing_coord = frame_coords[landing_frame]

    print(f"起跳点坐标: ({takeoff[0]:.2f}, {takeoff[1]:.2f}) "
          f"(帧 {takeoff_frame}, 对应坐标: ({takeoff_coord[0]:.2f}, {takeoff_coord[1]:.2f}))")
    print(f"落点坐标: ({landing[0]:.2f}, {landing[1]:.2f}) "
          f"(帧 {landing_frame}, 对应坐标: ({landing_coord[0]:.2f}, {landing_coord[1]:.2f}))")
    print(f"总帧数: {len(frame_coords)} "
          f"(帧率 {args.fps:g} fps, 1像素={args.pixel_to_meter:g}米)")

    # 调用速度分析函数
    speed_analysis = analyze_midair_speed(
        frame_coords, takeoff_frame, landing_frame, distance,
        fps=args.fps, pixel_to_meter=args.pixel_to_meter)

    print("\n=== 滞空阶段速度分析 ===")
    print(f"上升阶段帧范围: {min(speed_analysis['rising_frames'])} 到 {speed_analysis['peak_frame']}")
    if speed_analysis["falling_frames"]:
        print(f"下降阶段帧范围: {speed_analysis['peak_frame'] + 1} 到 "
              f"{max(speed_analysis['falling_frames'])}")
    else:
        print(f"下降阶段帧范围: 无（最高点就在落地帧 {speed_analysis['peak_frame']}）")
    print(f"滞空时长: {speed_analysis['duration_s']:.2f} 秒（{speed_analysis['fit_mode']}）")
    print(f"实际平均上升速度 (x, y): "
          f"({speed_analysis['avg_rising_speed'][0]:.2f}, "
          f"{speed_analysis['avg_rising_speed'][1]:.2f}) 米/秒")
    print(f"实际平均下降速度 (x, y): "
          f"({speed_analysis['avg_falling_speed'][0]:.2f}, "
          f"{speed_analysis['avg_falling_speed'][1]:.2f}) 米/秒")

    # 输出抛物线方程
    a, b, c = speed_analysis["parabola_coefficients"]
    print("\n=== 滞空阶段抛物线方程 ===")
    print(f"y = {a:.4f}x² + {b:.4f}x + {c:.4f}")

    if not args.no_show:
        plot_results(coords, labels, centers, takeoff, landing)
        print("图表已显示，关闭窗口可结束程序")


if __name__ == "__main__":
    main()
