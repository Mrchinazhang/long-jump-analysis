"""核心逻辑的回归测试，覆盖函数可导入性、参数化生效、退化情形与入参校验。

运行方式（不需要额外依赖，用标准库 unittest）：
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import long_jump_analysis as lj  # noqa: E402


def _frames(coords, clusters=lj.DEFAULT_CLUSTERS):
    """把坐标跑完聚类 + 插值，返回 (起跳帧, 落地帧, 重采样后坐标)。"""
    takeoff, landing, _, _, frame_coords = lj.analyze_coordinates(
        coords, clusters=clusters)
    tf = int(np.argmin(np.linalg.norm(frame_coords - takeoff, axis=1)))
    lf = int(np.argmin(np.linalg.norm(frame_coords - landing, axis=1)))
    return tf, lf, frame_coords


class TestPublicAPI(unittest.TestCase):
    def test_functions_are_module_level(self):
        """核心函数必须是模块级函数，能被 import 而不是埋在 __main__ 里。"""
        self.assertTrue(callable(lj.analyze_midair_speed))
        self.assertEqual(lj.analyze_midair_speed.__qualname__,
                         "analyze_midair_speed")
        self.assertTrue(callable(lj.analyze_coordinates))

    def test_defaults_exposed(self):
        """默认参数以模块常量暴露，便于调用方覆盖。"""
        for name in ("DEFAULT_FPS", "DEFAULT_PIXEL_TO_METER",
                     "DEFAULT_FRAMES", "DEFAULT_DISTANCE_REF_M"):
            self.assertIsInstance(getattr(lj, name), (int, float))


class TestDemoData(unittest.TestCase):
    def setUp(self):
        self.tf, self.lf, self.frame_coords = _frames(lj.demo_coordinates())

    def test_demo_runs(self):
        """内置示例数据能完整跑通，不抛 IndexError。"""
        result = lj.analyze_midair_speed(
            self.frame_coords, self.tf, self.lf, 0.0)
        self.assertGreaterEqual(result["peak_frame"], self.tf)
        self.assertLessEqual(result["peak_frame"], self.lf)
        self.assertIsInstance(result["avg_rising_speed"], np.ndarray)

    def test_demo_is_parabola(self):
        """示例数据应被识别为标准抛物线，且顶点落在中间。"""
        result = lj.analyze_midair_speed(
            self.frame_coords, self.tf, self.lf, 0.0)
        self.assertEqual(result["fit_mode"], "抛物线")
        self.assertTrue(result["falling_frames"])
        self.assertLess(self.tf, result["peak_frame"])
        self.assertLess(result["peak_frame"], self.lf)

    def test_rising_and_falling_are_symmetric(self):
        """对称的示例轨迹，上升与下降的纵向速度应大小相等、方向相反。

        顶点帧归入上升段，所以两段的速度样本数相差一个，平均值会有一个
        很小的离散误差，这里按 2% 容差比较。
        """
        result = lj.analyze_midair_speed(
            self.frame_coords, self.tf, self.lf, 0.0)
        rising_y = result["avg_rising_speed"][1]
        falling_y = result["avg_falling_speed"][1]
        self.assertGreater(rising_y, 0)
        self.assertLess(falling_y, 0)
        self.assertAlmostEqual(rising_y, -falling_y, delta=abs(rising_y) * 0.02)


class TestParameterization(unittest.TestCase):
    def setUp(self):
        self.tf, self.lf, self.frame_coords = _frames(lj.demo_coordinates())
        self.base = lj.analyze_midair_speed(
            self.frame_coords, self.tf, self.lf, 0.0)

    def test_fps_doubles_speed(self):
        """帧率翻倍：速度翻倍、滞空时长减半。"""
        result = lj.analyze_midair_speed(
            self.frame_coords, self.tf, self.lf, 0.0, fps=lj.DEFAULT_FPS * 2)
        ratio = (result["avg_rising_speed"][1]
                 / self.base["avg_rising_speed"][1])
        self.assertAlmostEqual(ratio, 2.0, places=9)
        self.assertAlmostEqual(result["duration_s"],
                               self.base["duration_s"] / 2, places=9)

    def test_pixel_to_meter_doubles_speed(self):
        """像素-米换算系数翻倍：速度翻倍。"""
        result = lj.analyze_midair_speed(
            self.frame_coords, self.tf, self.lf, 0.0,
            pixel_to_meter=lj.DEFAULT_PIXEL_TO_METER * 2)
        ratio = (result["avg_rising_speed"][1]
                 / self.base["avg_rising_speed"][1])
        self.assertAlmostEqual(ratio, 2.0, places=9)

    def test_distance_rescaling(self):
        """跳远成绩翻倍时速度按比例缩放，成绩为 0 时不做缩放。"""
        r10 = lj.analyze_midair_speed(self.frame_coords, self.tf, self.lf, 10.0)
        r20 = lj.analyze_midair_speed(self.frame_coords, self.tf, self.lf, 20.0)
        y10 = r10["avg_rising_speed"][1]
        y20 = r20["avg_rising_speed"][1]
        self.assertAlmostEqual(y20 / y10, 2.0, places=9)

        unscaled = lj.analyze_midair_speed(self.frame_coords, self.tf, self.lf, 0.0)
        expected = y10 / (10.0 / lj.DEFAULT_DISTANCE_REF_M)
        self.assertAlmostEqual(unscaled["avg_rising_speed"][1], expected, places=9)

    def test_frames_resampling(self):
        """重采样帧数可调，输出长度跟随参数变化。"""
        coords = lj.demo_coordinates()
        for frames in (2, 50, 300, 1000):
            *_, frame_coords = lj.analyze_coordinates(coords, frames=frames)
            self.assertEqual(len(frame_coords), frames)

    def test_no_duplicate_frames(self):
        """重采样是插值而不是索引抽样，相邻帧不应成片重复。"""
        *_, frame_coords = lj.analyze_coordinates(lj.demo_coordinates())
        steps = np.diff(frame_coords[:, 0])
        self.assertTrue(np.all(steps > 0), "x 方向应严格单调，不应出现零位移平台")


class TestDegenerateCases(unittest.TestCase):
    def test_collinear_data(self):
        """共线数据 a≈0：没有内部最高点，必须退回取最高帧而不是崩溃。"""
        line = np.array([[i, i] for i in range(1, 13)], dtype=float)
        tf, lf, frame_coords = _frames(line)
        result = lj.analyze_midair_speed(frame_coords, tf, lf, 0.0)
        self.assertIn("直线", result["fit_mode"])
        self.assertEqual(result["avg_falling_speed"].shape, (2,))

    def test_convex_data(self):
        """开口向上的拟合：顶点是最低点，不能当最高点用。"""
        xs = np.linspace(0, 10, 40)
        arc = np.column_stack([xs, 0.5 * (xs - 5) ** 2])
        tf, lf, frame_coords = _frames(arc)
        result = lj.analyze_midair_speed(frame_coords, tf, lf, 0.0)
        self.assertIn("开口向上", result["fit_mode"])

    def test_single_peak_at_landing(self):
        """顶点就是最后一帧时，下降段为空但不报错。"""
        xs = np.linspace(0, 10, 20)
        ramp = np.column_stack([xs, xs])
        tf, lf, frame_coords = _frames(ramp)
        result = lj.analyze_midair_speed(frame_coords, tf, lf, 0.0)
        if not result["falling_frames"]:
            self.assertEqual(result["avg_falling_speed"].tolist(), [0.0, 0.0])


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tf, self.lf, self.frame_coords = _frames(lj.demo_coordinates())

    def assertChineseValueError(self, func, **kwargs):
        with self.assertRaises(ValueError) as ctx:
            func(**kwargs)
        message = str(ctx.exception)
        self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in message),
                        f"错误信息应可读，实际为：{message}")

    def test_too_few_points(self):
        self.assertChineseValueError(
            lj.analyze_coordinates, coordinates=np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_bad_shape(self):
        self.assertChineseValueError(
            lj.analyze_coordinates, coordinates=np.array([1.0, 2.0, 3.0]))

    def test_frames_too_small(self):
        self.assertChineseValueError(
            lj.analyze_coordinates, coordinates=lj.demo_coordinates(), frames=1)

    def test_landing_before_takeoff(self):
        self.assertChineseValueError(
            lj.analyze_midair_speed,
            frame_coords=self.frame_coords, takeoff_frame=200, landing_frame=100)

    def test_bad_fps(self):
        self.assertChineseValueError(
            lj.analyze_midair_speed,
            frame_coords=self.frame_coords, takeoff_frame=self.tf,
            landing_frame=self.lf, fps=0)

    def test_bad_pixel_scale(self):
        self.assertChineseValueError(
            lj.analyze_midair_speed,
            frame_coords=self.frame_coords, takeoff_frame=self.tf,
            landing_frame=self.lf, pixel_to_meter=0)

    def test_midair_too_short(self):
        self.assertChineseValueError(
            lj.analyze_midair_speed,
            frame_coords=self.frame_coords, takeoff_frame=10, landing_frame=11)


class TestCommandLine(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = lj.parse_args([])
        self.assertEqual(args.fps, lj.DEFAULT_FPS)
        self.assertEqual(args.frames, lj.DEFAULT_FRAMES)
        self.assertFalse(args.demo)

    def test_parse_args_overrides(self):
        args = lj.parse_args(["--fps", "60", "--pixel-to-meter", "0.05",
                              "--frames", "120", "--clusters", "4",
                              "--demo", "--no-show"])
        self.assertEqual(args.fps, 60.0)
        self.assertEqual(args.pixel_to_meter, 0.05)
        self.assertEqual(args.frames, 120)
        self.assertEqual(args.clusters, 4)
        self.assertTrue(args.demo)
        self.assertTrue(args.no_show)

    def test_main_demo_runs(self):
        """--demo --no-show 应能完整跑完并正常返回。"""
        lj.main(["--demo", "--no-show"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
