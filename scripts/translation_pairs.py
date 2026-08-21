#!/usr/bin/env python3
"""生成“真实相机位移”Cross-Pair 的独立 20-case 演示页。

本脚本与现有 100-case 旋转项目互不覆盖，输出全部放在 ``site/translation/``。
它解决的是一个与旧管线不同的问题：真实位移不能由单帧 ERP 通过旋转矩阵生成，
所以 ref/target 必须读取同一条移动相机视频中的两个不同时间窗口。两个窗口的
起点尽量分开，使相机中心形成真实空间基线；两侧再共享同一条虚拟旋转轨迹。

输出分为两类：

1. ``translation_only``：10 组原生位移，不额外施加随时间变化的虚拟旋转。
2. ``translation_rotation``：10 组真实位移，并在两侧同步叠加 yaw/pitch/roll。

每侧严格 10 秒、20fps，像素数不超过 960×960，网页永远左右并排展示 ref/target。
三个源视频都使用原速 10 秒窗口，不循环、不冻结、不补黑帧，也不做时间拉伸。
"""

from __future__ import annotations

import argparse
import concurrent.futures
from html import escape
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# 复用主项目已经验证过的球面映射、编码器和元数据工具，避免维护第二套几何代码。
from pipeline import (  # noqa: E402
    RawFFmpegWriter,
    base_hfov_for_resolution,
    rectilinear_maps,
    rectilinear_rays,
    sampled_view_longitudes,
    sha256_file,
    video_metadata,
    write_json,
)


DEFAULT_SOURCE_DIR = Path("/Users/bytedance/Downloads/数据集")
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "site" / "translation"
MEDIA_ROOT = OUTPUT_ROOT / "media" / "cases"
MANIFEST_PATH = OUTPUT_ROOT / "CASE_MANIFEST.json"
VALIDATION_PATH = OUTPUT_ROOT / "VALIDATION.json"
INDEX_PATH = OUTPUT_ROOT / "index.html"

DURATION_SECONDS = 10.0
FPS = 20
MAX_PIXELS = 960 * 960
RESOLUTIONS = [
    [1280, 720],
    [960, 960],
    [720, 1280],
    [1104, 832],
    [832, 1104],
    [1440, 640],
    [640, 1440],
    [1344, 672],
    [1072, 856],
    [856, 1072],
]


def load_render_config() -> dict[str, Any]:
    """读取主项目配置，并为本页稍微提高编码质量。"""

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["render"] = {
        **config["render"],
        "crf": 20,
        "preset": "fast",
        "maxrate": "4000k",
        "bufsize": "8000k",
    }
    return config


class RetimeClipReader:
    """按给定源窗口读取，必要时对整段轨迹做统一时间缩放。

    ``source_span_seconds`` 是输出 10 秒实际覆盖的源时长。三个视频都取 10 秒，
    因此播放速度为 1.0，不会循环、停帧或改变原始相机速度。
    读取始终按源帧顺序前进，不会对每个输出帧反复 seek。
    """

    def __init__(
        self,
        path: Path,
        start_seconds: float,
        source_span_seconds: float,
        output_duration_seconds: float,
        output_fps: int,
    ) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开源视频：{path}")
        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.source_fps <= 0:
            raise RuntimeError(f"源视频 fps 无效：{path}")
        self.start_frame = int(round(start_seconds * self.source_fps))
        self.source_span_seconds = float(source_span_seconds)
        self.output_duration_seconds = float(output_duration_seconds)
        self.output_fps = int(output_fps)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.current_source_index = self.start_frame - 1
        self.current_frame: np.ndarray | None = None
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frame_for_output(self, output_index: int) -> np.ndarray:
        output_seconds = output_index / self.output_fps
        source_seconds = output_seconds * self.source_span_seconds / self.output_duration_seconds
        desired = self.start_frame + int(round(source_seconds * self.source_fps))
        while self.current_source_index < desired:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"源视频在窗口结束前解码失败：{self.path}")
            self.current_source_index += 1
            self.current_frame = frame
        if self.current_frame is None:
            raise RuntimeError(f"未读到首帧：{self.path}")
        return self.current_frame

    def close(self) -> None:
        self.capture.release()


def source_pair_windows(
    duration_seconds: float,
    source_span_seconds: float,
    count: int,
    edge_band_seconds: float,
) -> list[tuple[float, float]]:
    """在视频开头和末尾各取一组窗口，保证每对窗口尽量相隔较远。

    只在一个很窄的 ``edge_band_seconds`` 范围内改变起点：这样不同 case 不会
    完全复用相同切片，同时 ref/target 仍分别位于整条移动路径的两端。
    """

    latest = duration_seconds - source_span_seconds
    if latest < -0.03:
        raise ValueError(
            f"源视频 {duration_seconds:.3f}s 短于要求窗口 {source_span_seconds:.3f}s"
        )
    latest = max(0.0, latest)
    band = min(float(edge_band_seconds), latest / 2.0)
    early = np.linspace(0.0, band, count)
    late = np.linspace(max(0.0, latest - band), latest, count)
    return [(round(float(a), 3), round(float(b), 3)) for a, b in zip(early, late)]


def trajectory_spec(index: int) -> dict[str, Any]:
    """返回第 index 个“位移+旋转”轨迹参数。

    每条轨迹都至少两个旋转轴同时变化；幅度和速度不同，但 yaw 控制在安全范围，
    避免普通透视视角越过 ERP 左右接缝。两侧逐帧调用同一轨迹，因此虚拟旋转严格
    同步，pair 的额外差异只来自原始移动相机的真实时间/空间基线。
    """

    specs = [
        {"name": "快速右扫+上仰", "pattern": "sweep_right", "yaw_deg": 104.0, "pitch_deg": 34.0, "roll_deg": 0.0},
        {"name": "快速左扫+下压", "pattern": "sweep_left", "yaw_deg": 112.0, "pitch_deg": 30.0, "roll_deg": 0.0},
        {"name": "变速右甩+俯仰波", "pattern": "whip_right", "yaw_deg": 118.0, "pitch_deg": 26.0, "roll_deg": 0.0},
        {"name": "变速左甩+反向俯仰", "pattern": "whip_left", "yaw_deg": 120.0, "pitch_deg": 28.0, "roll_deg": 0.0},
        {"name": "蛇形摇摄", "pattern": "serpentine", "yaw_deg": 88.0, "pitch_deg": 38.0, "roll_deg": 0.0},
        {"name": "对角环绕", "pattern": "diagonal", "yaw_deg": 102.0, "pitch_deg": 48.0, "roll_deg": 0.0},
        {"name": "柔和螺旋", "pattern": "corkscrew", "yaw_deg": 92.0, "pitch_deg": 30.0, "roll_deg": 86.0},
        {"name": "大幅摆动+滚转", "pattern": "swing_roll", "yaw_deg": 90.0, "pitch_deg": 32.0, "roll_deg": 110.0},
        {"name": "俯仰穿越+轻滚", "pattern": "pitch_roll", "yaw_deg": 54.0, "pitch_deg": 64.0, "roll_deg": 76.0},
        {"name": "双脉冲甩镜", "pattern": "double_whip", "yaw_deg": 116.0, "pitch_deg": 36.0, "roll_deg": 0.0},
    ]
    return dict(specs[index])


def build_plan(source_dir: Path) -> dict[str, Any]:
    """生成固定、可复现的 20-case 计划。"""

    source_dir = source_dir.expanduser().resolve()
    source_rules = {
        "NSC.mp4": {"pair_count": 14, "source_span_seconds": 10.0, "edge_band_seconds": 10.0},
        "NSK.mp4": {"pair_count": 4, "source_span_seconds": 10.0, "edge_band_seconds": 0.6},
        "FTP.mp4": {"pair_count": 2, "source_span_seconds": 10.0, "edge_band_seconds": 0.2},
    }
    pools: dict[str, list[dict[str, Any]]] = {}
    source_summary: list[dict[str, Any]] = []
    for filename, rule in source_rules.items():
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        meta = video_metadata(path)
        if not meta.get("decode_ok"):
            raise RuntimeError(f"无法读取源视频：{path}")
        if (int(meta["width"]), int(meta["height"])) != (3840, 1920):
            raise ValueError(f"{filename} 不是预期的 3840×1920 ERP：{meta}")
        duration = float(meta["duration_seconds"])
        windows = source_pair_windows(
            duration,
            float(rule["source_span_seconds"]),
            int(rule["pair_count"]),
            float(rule["edge_band_seconds"]),
        )
        pools[filename] = [
            {
                "ref_start_seconds": ref_start,
                "target_start_seconds": target_start,
                "temporal_gap_seconds": round(abs(target_start - ref_start), 3),
                "source_span_seconds": float(rule["source_span_seconds"]),
            }
            for ref_start, target_start in windows
        ]
        source_summary.append(
            {
                "file": filename,
                "width": int(meta["width"]),
                "height": int(meta["height"]),
                "fps": round(float(meta["fps"]), 6),
                "duration_seconds": round(duration, 6),
                "source_span_seconds": float(rule["source_span_seconds"]),
                "output_speed_factor": round(
                    float(rule["source_span_seconds"]) / DURATION_SECONDS, 6
                ),
                "pair_count": int(rule["pair_count"]),
            }
        )

    # 两种类别都使用三个源；同一个池中的窗口只分配一次。
    pure_sources = ["NSC.mp4"] * 7 + ["NSK.mp4"] * 2 + ["FTP.mp4"]
    mixed_sources = ["NSC.mp4"] * 7 + ["NSK.mp4"] * 2 + ["FTP.mp4"]
    used: dict[str, int] = {name: 0 for name in pools}
    pure_yaws = [-104.0, -76.0, -46.0, -16.0, 16.0, 46.0, 76.0, 104.0, -58.0, 58.0]
    pure_pitches = [-10.0, -6.0, -2.0, 3.0, 7.0, 11.0, -11.0, 6.0, 0.0, -4.0]

    cases: list[dict[str, Any]] = []
    for index, filename in enumerate(pure_sources):
        pair = pools[filename][used[filename]]
        used[filename] += 1
        width, height = RESOLUTIONS[index]
        cases.append(
            {
                "id": f"translation_{index + 1:02d}",
                "kind": "translation_only",
                "display_name": "原生位移（未叠加虚拟旋转）",
                "source": {"dataset": "user_mobile_panorama", "file": filename},
                **pair,
                "duration_seconds": DURATION_SECONDS,
                "fps": FPS,
                "resolution": [width, height],
                "fixed_view": {
                    "yaw_deg": pure_yaws[index],
                    "pitch_deg": pure_pitches[index],
                    "roll_deg": 0.0,
                },
                "virtual_rotation": None,
            }
        )

    for index, filename in enumerate(mixed_sources):
        pair = pools[filename][used[filename]]
        used[filename] += 1
        width, height = RESOLUTIONS[index]
        cases.append(
            {
                "id": f"translation_rotation_{index + 1:02d}",
                "kind": "translation_rotation",
                "display_name": "真实位移 + 同步可控旋转",
                "source": {"dataset": "user_mobile_panorama", "file": filename},
                **pair,
                "duration_seconds": DURATION_SECONDS,
                "fps": FPS,
                "resolution": [width, height],
                "fixed_view": None,
                "virtual_rotation": trajectory_spec(index),
            }
        )

    plan = {
        "schema_version": 1,
        "case_count": len(cases),
        "rules": {
            "same_source_file_within_pair": True,
            "different_time_windows_supply_real_translation": True,
            "same_virtual_rotation_on_ref_and_target": True,
            "duration_seconds_each": DURATION_SECONDS,
            "fps": FPS,
            "max_pixels": MAX_PIXELS,
            "source_timing_note": "All three sources use distinct 10s windows at original speed without looping or retiming.",
        },
        "sources": source_summary,
        "cases": cases,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST_PATH, plan)
    return plan


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def whip(values: np.ndarray, sharpness: float = 6.5) -> np.ndarray:
    denominator = 2.0 * math.tanh(sharpness / 2.0)
    return (
        np.tanh(sharpness * (values - 0.5)) + math.tanh(sharpness / 2.0)
    ) / denominator


def trajectory_arrays(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """构造 300 帧相机方向，并生成接缝安全报告。"""

    frame_count = int(round(DURATION_SECONDS * FPS))
    p = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    width, height = map(int, case["resolution"])
    hfov = float(base_hfov_for_resolution(width, height, config["trajectory"]))

    if case["kind"] == "translation_only":
        view = case["fixed_view"]
        yaw = np.full(frame_count, float(view["yaw_deg"]), np.float32)
        pitch = np.full(frame_count, float(view["pitch_deg"]), np.float32)
        roll = np.full(frame_count, float(view["roll_deg"]), np.float32)
    else:
        spec = case["virtual_rotation"]
        yaw_amp = float(spec["yaw_deg"])
        pitch_amp = float(spec["pitch_deg"])
        roll_amp = float(spec["roll_deg"])
        pattern = spec["pattern"]
        if pattern == "sweep_right":
            yaw = yaw_amp * (p - 0.5)
            pitch = -pitch_amp * 0.5 + pitch_amp * smoothstep(p)
            roll = np.zeros_like(p)
        elif pattern == "sweep_left":
            yaw = yaw_amp * (0.5 - p)
            pitch = pitch_amp * 0.5 - pitch_amp * smoothstep(p)
            roll = np.zeros_like(p)
        elif pattern == "whip_right":
            yaw = yaw_amp * (whip(p, 7.8) - 0.5)
            pitch = pitch_amp * np.sin(np.pi * p)
            roll = np.zeros_like(p)
        elif pattern == "whip_left":
            yaw = yaw_amp * (0.5 - whip(p, 8.5))
            pitch = -pitch_amp * np.sin(np.pi * p)
            roll = np.zeros_like(p)
        elif pattern == "serpentine":
            yaw = yaw_amp * 0.5 * np.sin(2.0 * np.pi * p)
            pitch = pitch_amp * 0.5 * np.sin(4.0 * np.pi * p + 0.4)
            roll = np.zeros_like(p)
        elif pattern == "diagonal":
            yaw = yaw_amp * (p - 0.5)
            pitch = pitch_amp * (0.5 - p) + 7.0 * np.sin(2.0 * np.pi * p)
            roll = np.zeros_like(p)
        elif pattern == "corkscrew":
            yaw = yaw_amp * (p - 0.5)
            pitch = pitch_amp * 0.5 * np.sin(2.0 * np.pi * p)
            roll = roll_amp * (p - 0.5)
        elif pattern == "swing_roll":
            yaw = yaw_amp * 0.5 * np.sin(2.0 * np.pi * p)
            pitch = pitch_amp * 0.5 * np.sin(3.0 * np.pi * p)
            roll = roll_amp * (p - 0.5)
        elif pattern == "pitch_roll":
            yaw = yaw_amp * 0.5 * np.sin(2.0 * np.pi * p)
            pitch = pitch_amp * (p - 0.5)
            roll = roll_amp * 0.5 * np.sin(2.0 * np.pi * p + 0.7)
        elif pattern == "double_whip":
            first = whip(np.clip(p * 2.0, 0.0, 1.0), 8.0)
            second = whip(np.clip((p - 0.5) * 2.0, 0.0, 1.0), 8.0)
            yaw = yaw_amp * 0.5 * (first - second)
            pitch = pitch_amp * 0.5 * np.sin(2.0 * np.pi * p)
            roll = np.zeros_like(p)
        else:
            raise ValueError(f"未知轨迹：{pattern}")
        yaw = yaw.astype(np.float32)
        pitch = pitch.astype(np.float32)
        roll = roll.astype(np.float32)

    longitude_limit = 180.0 - float(config["trajectory"]["seam_margin_deg"])
    max_abs_longitude = 0.0
    for frame_index in list(range(0, frame_count, 4)) + [frame_count - 1]:
        longitudes = sampled_view_longitudes(
            width,
            height,
            hfov,
            float(yaw[frame_index]),
            float(pitch[frame_index]),
            float(roll[frame_index]),
        )
        max_abs_longitude = max(max_abs_longitude, float(np.max(np.abs(longitudes))))
    if max_abs_longitude > longitude_limit:
        raise RuntimeError(
            f"{case['id']} 轨迹触及 ERP 接缝：{max_abs_longitude:.2f}>{longitude_limit:.2f}"
        )

    angular_speed = np.linalg.norm(
        np.stack([np.diff(yaw), np.diff(pitch), np.diff(roll)], axis=1), axis=1
    ) * FPS
    return {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "hfov_deg": hfov,
        "report": {
            "yaw_range_deg": [round(float(yaw.min()), 4), round(float(yaw.max()), 4)],
            "pitch_range_deg": [round(float(pitch.min()), 4), round(float(pitch.max()), 4)],
            "roll_range_deg": [round(float(roll.min()), 4), round(float(roll.max()), 4)],
            "hfov_deg": round(hfov, 4),
            "max_abs_sampled_longitude_deg": round(max_abs_longitude, 4),
            "seam_limit_deg": round(longitude_limit, 4),
            "seam_safe": True,
            "virtual_rotation_median_deg_per_second": round(
                float(np.median(angular_speed)), 4
            ),
            "virtual_rotation_p99_deg_per_second": round(
                float(np.percentile(angular_speed, 99)), 4
            ),
        },
    }


def make_poster(video_path: Path, poster_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"无法生成预览图：{video_path}")
    scale = min(1.0, 720.0 / max(frame.shape[1], frame.shape[0]))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if not cv2.imwrite(str(poster_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"写入预览图失败：{poster_path}")


def render_case(
    case: dict[str, Any],
    source_dir: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    case_dir = MEDIA_ROOT / case["id"]
    ref_path = case_dir / "ref.mp4"
    target_path = case_dir / "target.mp4"
    metadata_path = case_dir / "metadata.json"
    if not overwrite and ref_path.exists() and target_path.exists() and metadata_path.exists():
        return {"id": case["id"], "status": "skipped_existing", "bytes": 0}

    source_path = source_dir / case["source"]["file"]
    width, height = map(int, case["resolution"])
    frame_count = int(round(DURATION_SECONDS * FPS))
    trajectory = trajectory_arrays(case, config)
    readers = {
        "ref": RetimeClipReader(
            source_path,
            float(case["ref_start_seconds"]),
            float(case["source_span_seconds"]),
            DURATION_SECONDS,
            FPS,
        ),
        "target": RetimeClipReader(
            source_path,
            float(case["target_start_seconds"]),
            float(case["source_span_seconds"]),
            DURATION_SECONDS,
            FPS,
        ),
    }
    writers = {
        "ref": RawFFmpegWriter(ref_path, width, height, FPS, config),
        "target": RawFFmpegWriter(target_path, width, height, FPS, config),
    }

    rays = rectilinear_rays(width, height, float(trajectory["hfov_deg"]))
    fixed_maps: tuple[np.ndarray, np.ndarray] | None = None
    try:
        for frame_index in range(frame_count):
            yaw = float(trajectory["yaw_deg"][frame_index])
            pitch = float(trajectory["pitch_deg"][frame_index])
            roll = float(trajectory["roll_deg"][frame_index])
            if case["kind"] == "translation_only" and fixed_maps is not None:
                maps = fixed_maps
            else:
                maps = rectilinear_maps(
                    rays,
                    (width, height),
                    (readers["ref"].width, readers["ref"].height),
                    yaw,
                    pitch,
                    roll,
                )
                if case["kind"] == "translation_only":
                    fixed_maps = maps

            for side in ("ref", "target"):
                source_frame = readers[side].frame_for_output(frame_index)
                output_frame = cv2.remap(
                    source_frame,
                    maps[0],
                    maps[1],
                    cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                writers[side].write(output_frame)
        writers["ref"].close(True)
        writers["target"].close(True)
    except Exception:
        for writer in writers.values():
            try:
                writer.close(False)
            except Exception:
                pass
        ref_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)
        raise
    finally:
        for reader in readers.values():
            reader.close()

    make_poster(ref_path, case_dir / "ref.jpg")
    make_poster(target_path, case_dir / "target.jpg")
    metadata = {
        "case": case,
        "trajectory_report": trajectory["report"],
        "rendered": {
            "ref": {
                "path": str(ref_path.relative_to(PROJECT_ROOT)),
                "bytes": ref_path.stat().st_size,
                "sha256": sha256_file(ref_path),
            },
            "target": {
                "path": str(target_path.relative_to(PROJECT_ROOT)),
                "bytes": target_path.stat().st_size,
                "sha256": sha256_file(target_path),
            },
            "codec": config["render"],
        },
    }
    write_json(metadata_path, metadata)
    return {
        "id": case["id"],
        "status": "rendered",
        "bytes": ref_path.stat().st_size + target_path.stat().st_size,
    }


def command_render(
    plan: dict[str, Any],
    source_dir: Path,
    workers: int,
    overwrite: bool,
    limit: int | None,
) -> None:
    config = load_render_config()
    cases = list(plan["cases"])
    if limit is not None:
        cases = cases[: int(limit)]
    payloads = [(case, source_dir, config, overwrite) for case in cases]

    if workers <= 1:
        results = []
        for index, payload in enumerate(payloads, 1):
            result = render_case(*payload)
            results.append(result)
            print(f"[{index}/{len(payloads)}] {result['id']}: {result['status']}", flush=True)
    else:
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(render_case, *payload) for payload in payloads]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(payloads)}] {result['id']}: {result['status']}", flush=True)
    total_bytes = sum(int(item["bytes"]) for item in results)
    print(f"渲染完成：{len(results)} cases，新增约 {total_bytes / 1024**2:.1f} MiB")


def command_validate(plan: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    file_rows: list[dict[str, Any]] = []
    cases = plan["cases"]
    if len(cases) != 20:
        errors.append(f"case 数量不是 20：{len(cases)}")
    if sum(case["kind"] == "translation_only" for case in cases) != 10:
        errors.append("纯位移 case 数量不是 10")
    if sum(case["kind"] == "translation_rotation" for case in cases) != 10:
        errors.append("位移+旋转 case 数量不是 10")

    config = load_render_config()
    for case in cases:
        width, height = map(int, case["resolution"])
        if width * height > MAX_PIXELS:
            errors.append(f"{case['id']} 分辨率超过 max pixels")
        source_path = source_dir / case["source"]["file"]
        source_meta = video_metadata(source_path)
        latest = float(source_meta.get("duration_seconds", 0.0)) - float(
            case["source_span_seconds"]
        )
        gap = float(case["temporal_gap_seconds"])
        if latest > 0 and gap < latest * 0.65:
            errors.append(f"{case['id']} 时间基线不够远：{gap:.3f}/{latest:.3f}s")

        trajectory = trajectory_arrays(case, config)
        if not trajectory["report"]["seam_safe"]:
            errors.append(f"{case['id']} 轨迹可能跨接缝")
        if case["kind"] == "translation_only":
            if trajectory["report"]["virtual_rotation_p99_deg_per_second"] > 0.001:
                errors.append(f"{case['id']} 意外含有虚拟旋转")
        elif trajectory["report"]["virtual_rotation_median_deg_per_second"] <= 2.0:
            errors.append(f"{case['id']} 虚拟旋转过弱")

        pair_hashes: list[str] = []
        for side in ("ref", "target"):
            path = MEDIA_ROOT / case["id"] / f"{side}.mp4"
            if not path.exists():
                errors.append(f"缺少 {path.relative_to(PROJECT_ROOT)}")
                continue
            meta = video_metadata(path)
            if not meta.get("decode_ok"):
                errors.append(f"无法解码 {path.relative_to(PROJECT_ROOT)}")
                continue
            if (int(meta["width"]), int(meta["height"])) != (width, height):
                errors.append(f"{case['id']} {side} 分辨率错误")
            if abs(float(meta["fps"]) - FPS) > 0.05:
                errors.append(f"{case['id']} {side} fps 错误")
            if abs(float(meta["duration_seconds"]) - DURATION_SECONDS) > 0.06:
                errors.append(f"{case['id']} {side} 时长不是 10 秒")
            digest = sha256_file(path)
            pair_hashes.append(digest)
            file_rows.append(
                {
                    "case_id": case["id"],
                    "side": side,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "width": int(meta["width"]),
                    "height": int(meta["height"]),
                    "pixels": int(meta["width"]) * int(meta["height"]),
                    "fps": round(float(meta["fps"]), 6),
                    "duration_seconds": round(float(meta["duration_seconds"]), 6),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        if len(pair_hashes) == 2 and pair_hashes[0] == pair_hashes[1]:
            errors.append(f"{case['id']} ref/target 内容完全相同")

    validation = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "case_count": len(cases),
        "video_count": len(file_rows),
        "counts": {
            "translation_only": sum(case["kind"] == "translation_only" for case in cases),
            "translation_rotation": sum(
                case["kind"] == "translation_rotation" for case in cases
            ),
        },
        "rules": plan["rules"],
        "total_video_bytes": sum(row["bytes"] for row in file_rows),
        "files": file_rows,
    }
    write_json(VALIDATION_PATH, validation)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise RuntimeError(f"验证失败：{len(errors)} 项")
    print("验证通过：20 cases / 40 MP4 / 每侧严格 10 秒")
    return validation


def video_panel(case: dict[str, Any], side: str) -> str:
    start = float(case[f"{side}_start_seconds"])
    label = "REF · 参考轨迹" if side == "ref" else "TARGET · 目标轨迹"
    return f"""
      <figure class="video-panel">
        <figcaption><b>{label}</b><span>{escape(case['source']['file'])}</span></figcaption>
        <video controls playsinline preload="metadata" src="media/cases/{case['id']}/{side}.mp4" poster="media/cases/{case['id']}/{side}.jpg" style="aspect-ratio:{case['resolution'][0]}/{case['resolution'][1]}"></video>
        <div class="source-line">源起点 {start:.2f}s · 覆盖 {float(case['source_span_seconds']):.1f}s</div>
      </figure>"""


def case_card(case: dict[str, Any]) -> str:
    width, height = case["resolution"]
    if case["kind"] == "translation_only":
        motion = "原生位移 · 无额外虚拟旋转"
        detail = (
            f"固定观察方向 yaw {float(case['fixed_view']['yaw_deg']):.0f}° / "
            f"pitch {float(case['fixed_view']['pitch_deg']):.0f}°"
        )
        css_class = "translation-only"
    else:
        rotation = case["virtual_rotation"]
        motion = f"真实位移 + {escape(rotation['name'])}"
        detail = (
            f"yaw {float(rotation['yaw_deg']):.0f}° · pitch {float(rotation['pitch_deg']):.0f}°"
            + (f" · roll {float(rotation['roll_deg']):.0f}°" if float(rotation["roll_deg"]) else "")
        )
        css_class = "translation-rotation"
    return f"""
    <article id="{case['id']}" class="case-card {css_class}" data-kind="{case['kind']}">
      <header>
        <div><span class="case-id">{case['id']}</span><h3>{motion}</h3></div>
        <div class="badges"><b>10.00s</b><b>{width}×{height}</b><b>时间基线 {float(case['temporal_gap_seconds']):.2f}s</b></div>
      </header>
      <p class="actions">{detail}</p>
      <div class="video-pair" data-sync-pair>
        {video_panel(case, 'ref')}
        {video_panel(case, 'target')}
      </div>
    </article>"""


def build_site(plan: dict[str, Any]) -> None:
    pure = [case for case in plan["cases"] if case["kind"] == "translation_only"]
    mixed = [case for case in plan["cases"] if case["kind"] == "translation_rotation"]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>真实位移全景 Cross-Pair Demo</title>
  <style>
    :root{{--ink:#17211f;--muted:#62706b;--paper:#f4f1e9;--card:#fffefb;--line:#d8d5ca;--green:#1d684f;--blue:#2b5f8d}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    main{{max-width:1320px;margin:auto;padding:22px 24px 70px}} .hero{{padding:18px 22px;border:1px solid var(--line);border-radius:18px;background:var(--card)}}
    h1{{font-size:clamp(24px,3vw,38px);line-height:1.2;margin:0}} h2,h3,p{{margin-top:0}}
    .filters{{position:sticky;top:0;z-index:8;display:flex;gap:8px;margin:18px 0;padding:10px;border:1px solid var(--line);border-radius:14px;background:rgba(244,241,233,.96)}}
    button{{border:1px solid #b9c1bd;border-radius:999px;background:white;padding:8px 13px;cursor:pointer}} button.active{{background:var(--ink);color:white;border-color:var(--ink)}}
    .group{{margin:28px 0}} .group-title{{font-size:27px;margin:0 0 12px}}
    .case-card{{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--blue);border-radius:18px;padding:18px;margin:16px 0;box-shadow:0 8px 22px rgba(33,42,38,.05)}}
    .case-card.translation-rotation{{border-left-color:var(--green)}} .case-card[hidden]{{display:none}}
    .case-card>header{{display:flex;justify-content:space-between;gap:18px;align-items:start}} .case-card h3{{margin:2px 0 4px;font-size:21px}} .case-id{{font:700 12px ui-monospace,monospace;color:var(--muted)}}
    .badges{{display:flex;gap:7px;flex-wrap:wrap;justify-content:end}} .badges b{{font-size:12px;background:#e8ede9;border-radius:999px;padding:5px 9px}} .actions{{font-weight:650;color:#34453f}}
    .video-pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;min-width:0}} .video-panel{{margin:0;min-width:0;background:#101312;border-radius:13px;overflow:hidden}}
    .video-panel figcaption{{display:flex;justify-content:space-between;padding:9px 11px;color:white}} .video-panel video{{display:block;width:100%;background:#000;object-fit:contain}}
    .source-line{{color:#d2d9d5;padding:8px 11px;font-size:12px}}
    @media(max-width:800px){{main{{padding:12px 10px 50px}}.case-card{{padding:10px;overflow-x:auto}}.video-pair{{min-width:720px}}.case-card>header{{display:block}}.badges{{justify-content:start;margin:8px 0}}}}
  </style>
</head>
<body><main>
  <section class="hero"><h1>20 个 10 秒真实位移 Cross-Pair：10 个原生位移，10 个真实位移 + 可控旋转</h1></section>
  <div class="filters"><button class="active" data-filter="all">全部 20</button><button data-filter="translation_only">原生位移 10</button><button data-filter="translation_rotation">位移 + 旋转 10</button></div>
  <section class="group" data-group="translation_only"><h2 class="group-title">原生位移</h2>{''.join(case_card(case) for case in pure)}</section>
  <section class="group" data-group="translation_rotation"><h2 class="group-title">真实位移 + 可控旋转</h2>{''.join(case_card(case) for case in mixed)}</section>
</main>
<script>
  document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
    document.querySelectorAll('[data-filter]').forEach(item => item.classList.toggle('active', item === button));
    const wanted = button.dataset.filter;
    document.querySelectorAll('[data-kind]').forEach(card => card.hidden = wanted !== 'all' && card.dataset.kind !== wanted);
    document.querySelectorAll('[data-group]').forEach(group => group.hidden = wanted !== 'all' && group.dataset.group !== wanted);
  }}));
  document.querySelectorAll('[data-sync-pair]').forEach(pair => {{
    const videos = [...pair.querySelectorAll('video')]; let syncing = false; let leader = videos[0]; let raf = null;
    const mirror = (source, eventName) => {{ if (syncing) return; syncing = true; videos.filter(v => v !== source).forEach(v => {{ if (Math.abs(v.currentTime-source.currentTime)>0.05) v.currentTime=source.currentTime; v.playbackRate=source.playbackRate; if(eventName==='play')v.play().catch(()=>{{}}); if(eventName==='pause')v.pause(); }}); requestAnimationFrame(()=>syncing=false); }};
    const align = () => {{ if (!leader || leader.paused) {{ raf=null; return; }} syncing=true; videos.filter(v=>v!==leader).forEach(v=>{{if(Math.abs(v.currentTime-leader.currentTime)>0.05)v.currentTime=leader.currentTime;v.playbackRate=leader.playbackRate;}}); syncing=false; raf=requestAnimationFrame(align); }};
    videos.forEach(video => {{ ['pointerdown','keydown'].forEach(name=>video.addEventListener(name,()=>leader=video)); ['play','pause','seeking','ratechange'].forEach(name=>video.addEventListener(name,()=>{{if(syncing)return;mirror(video,name);if(name==='play'&&raf===null)raf=requestAnimationFrame(align);}})); }});
  }});
</script></body></html>"""
    # 模板为了可读性包含缩进空行；发布前统一去掉行尾空格，保证 git diff --check
    # 和静态站点构建日志保持干净。
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    INDEX_PATH.write_text(html, encoding="utf-8")
    print(f"网页已生成：{INDEX_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成真实位移全景 Cross-Pair Demo")
    parser.add_argument("command", choices=["plan", "render", "validate", "site", "all"])
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    if args.command in {"plan", "all"} or not MANIFEST_PATH.exists():
        plan = build_plan(source_dir)
        print(f"计划已生成：{MANIFEST_PATH}")
    else:
        plan = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if args.command in {"render", "all"}:
        command_render(plan, source_dir, max(1, int(args.workers)), args.overwrite, args.limit)
    if args.command in {"validate", "all"}:
        command_validate(plan, source_dir)
    if args.command in {"site", "all"}:
        build_site(plan)


if __name__ == "__main__":
    main()
