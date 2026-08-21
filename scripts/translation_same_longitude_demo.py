#!/usr/bin/env python3
"""同经度上下双取景框的纯二维移动全景 Cross-Pair 演示。

这个脚本是独立实验，不修改 ``pipeline.py`` 或现有网页。每个 pair 只创建一个源
视频 reader：ref/target 在完全相同的 10 秒、完全相同的源帧上采样。两个透视框
共享同一个 yaw（ERP 经度）和同一个透视母平面外参，只在母平面中截取上下两个
偏轴窗口。两个窗口都位于投影主点下方，因此上下分离、不重叠，也避免天空空镜。

输出两组案例：

* 10 个原生移动相机 case：不额外增加虚拟旋转；
* 10 个原生移动相机 + 公共旋转 case：两侧逐帧 yaw/pitch/roll 增量完全一致。

共享母平面让两侧的相机外参、左右轴和焦距一致，并避免“左右对称经度导致横向
方向翻转”。两个偏轴窗口位于主点同侧，也让前进/后退造成的径向方向保持同号。
不同区域的真实深度仍可能让像流幅度略有差异；本脚本不做镜像或稳定化。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
from html import escape
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline import (  # noqa: E402
    RawFFmpegWriter,
    rectilinear_maps,
    rotation_matrix,
    sha256_file,
    video_metadata,
    write_json,
)


DEFAULT_SOURCE_DIR = Path("/Users/bytedance/Downloads/数据集")
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "site" / "translation-same-longitude"
MEDIA_ROOT = OUTPUT_ROOT / "media" / "cases"
MANIFEST_PATH = OUTPUT_ROOT / "CASE_MANIFEST.json"
VALIDATION_PATH = OUTPUT_ROOT / "VALIDATION.json"
INDEX_PATH = OUTPUT_ROOT / "index.html"

DURATION_SECONDS = 10.0
FPS = 20
FRAME_COUNT = int(DURATION_SECONDS * FPS)
MAX_PIXELS = 960 * 960
SOURCE_SPEED_FACTOR = 0.5
SOURCE_SPAN_SECONDS = DURATION_SECONDS * SOURCE_SPEED_FACTOR
OUTPUT_HFOV_DEG = 80.0
MASTER_PRINCIPAL_Y_RATIO = -0.15
CROP_GAP_RATIO = 0.32
RANDOM_SEED = 20260822

RESOLUTIONS = [
    [1280, 720],
    [1440, 640],
    [1344, 672],
    [1536, 600],
    [1600, 576],
] * 4

PURE_YAWS = [-70.0, -55.0, -40.0, -25.0, -10.0, 10.0, 25.0, 40.0, 55.0, 70.0]
COMBINED_YAWS = [-65.0, -50.0, -35.0, -20.0, -5.0, 5.0, 20.0, 35.0, 50.0, 65.0]


def load_render_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["render"] = {
        **config["render"],
        "crf": 18,
        "preset": "fast",
        "maxrate": "6000k",
        "bufsize": "12000k",
    }
    return config


class SharedClipReader:
    """pair 内唯一的源 reader，确保两侧永远使用同一个源帧索引。"""

    def __init__(self, path: Path, start_seconds: float, speed_factor: float) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开源视频：{path}")
        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.source_fps <= 0:
            raise RuntimeError(f"源 fps 无效：{path}")
        self.start_frame = int(round(start_seconds * self.source_fps))
        self.speed_factor = float(speed_factor)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.current_source_index = self.start_frame - 1
        self.current_frame: np.ndarray | None = None
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frame(self, output_index: int) -> np.ndarray:
        desired = self.start_frame + int(
            round(output_index * self.source_fps / FPS * self.speed_factor)
        )
        while self.current_source_index < desired:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"源视频提前结束：{self.path}")
            self.current_source_index += 1
            self.current_frame = frame
        if self.current_frame is None:
            raise RuntimeError(f"没有读到源帧：{self.path}")
        return self.current_frame

    def close(self) -> None:
        self.capture.release()


def evenly_spaced_starts(duration: float, count: int) -> list[float]:
    latest = duration - SOURCE_SPAN_SECONDS
    if latest < -0.03:
        raise ValueError(f"源视频不足 {SOURCE_SPAN_SECONDS:.1f} 秒：{duration:.3f}s")
    return [round(float(value), 3) for value in np.linspace(0.0, max(0.0, latest), count)]


def rotation_spec(index: int, rng: random.Random) -> dict[str, Any]:
    specs = [
        ("持续右摇+轻仰", "linear", 20.0, 6.0, 0.0),
        ("持续左摇+轻俯", "linear_reverse", 22.0, 7.0, 0.0),
        ("快速右甩", "whip", 24.0, 5.0, 0.0),
        ("快速左甩", "whip_reverse", 24.0, 6.0, 0.0),
        ("蛇形摇摄", "wave", 20.0, 8.0, 0.0),
        ("对角移动", "diagonal", 18.0, 10.0, 0.0),
        ("轻螺旋", "corkscrew", 18.0, 7.0, 18.0),
        ("摆动+滚转", "swing_roll", 20.0, 8.0, 20.0),
        ("俯仰+轻滚", "pitch_roll", 14.0, 12.0, 16.0),
        ("双脉冲甩镜", "double_whip", 22.0, 7.0, 0.0),
    ]
    name, pattern, yaw, pitch, roll = specs[index]
    speed_multiplier = round(rng.uniform(0.55, 1.45), 3)
    return {
        "name": name,
        "pattern": pattern,
        "yaw_amplitude_deg": yaw,
        "pitch_amplitude_deg": pitch,
        "roll_amplitude_deg": roll,
        "speed_multiplier": speed_multiplier,
    }


def build_plan(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    rng = random.Random(RANDOM_SEED)
    source_rules = {"NSC.mp4": 14, "NSK.mp4": 4, "FTP.mp4": 2}
    pools: dict[str, list[float]] = {}
    sources: list[dict[str, Any]] = []
    for filename, count in source_rules.items():
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        meta = video_metadata(path)
        if not meta.get("decode_ok"):
            raise RuntimeError(path)
        if (int(meta["width"]), int(meta["height"])) != (3840, 1920):
            raise ValueError(f"{filename} 不是 3840×1920 ERP：{meta}")
        pools[filename] = evenly_spaced_starts(float(meta["duration_seconds"]), count)
        sources.append(
            {
                "file": filename,
                "width": int(meta["width"]),
                "height": int(meta["height"]),
                "fps": round(float(meta["fps"]), 6),
                "duration_seconds": round(float(meta["duration_seconds"]), 6),
                "case_count": count,
            }
        )

    pure_sources = ["NSC.mp4"] * 7 + ["NSK.mp4"] * 2 + ["FTP.mp4"]
    combined_sources = ["NSC.mp4"] * 7 + ["NSK.mp4"] * 2 + ["FTP.mp4"]
    used = {name: 0 for name in pools}
    pure_upper_roles = ["target"] * 7 + ["ref"] * 3
    combined_upper_roles = ["target"] * 8 + ["ref"] * 2
    rng.shuffle(pure_upper_roles)
    rng.shuffle(combined_upper_roles)
    upper_roles = pure_upper_roles + combined_upper_roles
    cases: list[dict[str, Any]] = []
    for combined, sequence in ((False, pure_sources), (True, combined_sources)):
        for index, filename in enumerate(sequence):
            start = pools[filename][used[filename]]
            used[filename] += 1
            case_id = (
                f"same_longitude_rotation_{index + 1:02d}"
                if combined
                else f"same_longitude_{index + 1:02d}"
            )
            yaw = COMBINED_YAWS[index] if combined else PURE_YAWS[index]
            width, height = RESOLUTIONS[len(cases)]
            # 每个 case 独立随机上下位置。两个等尺寸窗口在共同母平面内严格不重叠，
            # 同时限制最下缘视角，避免逼近 ERP 南极造成极端拉伸。
            for _ in range(100):
                principal_ratio = rng.uniform(-0.10, 0.02)
                upper_origin_ratio = rng.uniform(0.02, 0.18)
                gap_ratio = rng.uniform(0.12, 0.30)
                focal = width / (2.0 * math.tan(math.radians(OUTPUT_HFOV_DEG) / 2.0))
                principal_y = height * principal_ratio
                upper_origin = height * upper_origin_ratio
                lower_origin = upper_origin + height * (1.0 + gap_ratio)
                upper_bottom = math.degrees(
                    math.atan2(upper_origin + height - 0.5 - principal_y, focal)
                )
                lower_top = math.degrees(
                    math.atan2(lower_origin + 0.5 - principal_y, focal)
                )
                lower_bottom = math.degrees(
                    math.atan2(lower_origin + height - 0.5 - principal_y, focal)
                )
                upper_top = math.degrees(
                    math.atan2(upper_origin + 0.5 - principal_y, focal)
                )
                if lower_top - upper_bottom > 3.0 and lower_bottom < 72.0 and upper_top > 0.0:
                    break
            else:
                raise RuntimeError(f"无法为 {case_id} 采样安全上下窗口")
            upper_side = upper_roles[len(cases)]
            cases.append(
                {
                    "id": case_id,
                    "kind": "translation_rotation" if combined else "translation_only",
                    "source": {"dataset": "user_mobile_panorama", "file": filename},
                    "start_seconds": start,
                    "ref_start_seconds": start,
                    "target_start_seconds": start,
                    "duration_seconds": DURATION_SECONDS,
                    "source_span_seconds": SOURCE_SPAN_SECONDS,
                    "source_speed_factor": SOURCE_SPEED_FACTOR,
                    "fps": FPS,
                    "resolution": [width, height],
                    "shared_yaw_deg": yaw,
                    "ref_view": {"yaw_deg": yaw, "pitch_deg": 0.0, "roll_deg": 0.0},
                    "target_view": {"yaw_deg": yaw, "pitch_deg": 0.0, "roll_deg": 0.0},
                    "off_axis_layout": {
                        "master_principal_y_ratio": round(principal_ratio, 6),
                        "upper_origin_y_ratio": round(upper_origin_ratio, 6),
                        "crop_gap_ratio": round(gap_ratio, 6),
                        "upper_side": upper_side,
                        "ref_crop": "upper" if upper_side == "ref" else "lower",
                        "target_crop": "upper" if upper_side == "target" else "lower",
                    },
                    "virtual_rotation": rotation_spec(index, rng) if combined else None,
                }
            )

    plan = {
        "schema_version": 1,
        "case_count": len(cases),
        "rules": {
            "two_dimensional_erp_projection_only": True,
            "same_source_time_and_frame_sequence": True,
            "same_longitude_yaw_within_pair": True,
            "shared_perspective_master_plane": True,
            "upper_lower_off_axis_windows": True,
            "no_mirror": True,
            "no_stabilization_or_motion_transfer": True,
            "same_added_rotation_delta_within_pair": True,
            "baseline_source_speed_factor": SOURCE_SPEED_FACTOR,
            "rotation_speed_randomized_per_combined_case": True,
            "upper_window_assignment": "random with 75% TARGET bias",
            "crop_vertical_position_randomized_per_case": True,
            "output_hfov_deg": OUTPUT_HFOV_DEG,
            "forward_backward_projection_may_differ": True,
            "duration_seconds": DURATION_SECONDS,
            "fps": FPS,
            "max_pixels": MAX_PIXELS,
        },
        "sources": sources,
        "cases": cases,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST_PATH, plan)
    return plan


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def whip(values: np.ndarray, sharpness: float = 7.0) -> np.ndarray:
    denominator = 2.0 * math.tanh(sharpness / 2.0)
    return (
        np.tanh(sharpness * (values - 0.5)) + math.tanh(sharpness / 2.0)
    ) / denominator


def common_rotation(case: dict[str, Any]) -> dict[str, Any]:
    p = np.linspace(0.0, 1.0, FRAME_COUNT, dtype=np.float32)
    if case["kind"] == "translation_only":
        yaw = np.zeros(FRAME_COUNT, np.float32)
        pitch = np.zeros(FRAME_COUNT, np.float32)
        roll = np.zeros(FRAME_COUNT, np.float32)
    else:
        spec = case["virtual_rotation"]
        speed = float(spec["speed_multiplier"])
        # 固定 10 秒内通过缩放总角位移改变平均角速度；两侧仍共享同一数组。
        ya = float(spec["yaw_amplitude_deg"]) * speed
        pa = float(spec["pitch_amplitude_deg"]) * speed
        ra = float(spec["roll_amplitude_deg"]) * speed
        pattern = spec["pattern"]
        if pattern == "linear":
            yaw, pitch, roll = ya * (p - 0.5), pa * (p - 0.5), np.zeros_like(p)
        elif pattern == "linear_reverse":
            yaw, pitch, roll = ya * (0.5 - p), pa * (0.5 - p), np.zeros_like(p)
        elif pattern == "whip":
            yaw, pitch, roll = ya * (whip(p) - 0.5), pa * np.sin(np.pi * p), np.zeros_like(p)
        elif pattern == "whip_reverse":
            yaw, pitch, roll = ya * (0.5 - whip(p)), -pa * np.sin(np.pi * p), np.zeros_like(p)
        elif pattern == "wave":
            yaw, pitch, roll = ya * 0.5 * np.sin(2 * np.pi * p), pa * 0.5 * np.sin(4 * np.pi * p), np.zeros_like(p)
        elif pattern == "diagonal":
            yaw, pitch, roll = ya * (p - 0.5), pa * (0.5 - p), np.zeros_like(p)
        elif pattern == "corkscrew":
            yaw, pitch, roll = ya * (p - 0.5), pa * 0.5 * np.sin(2 * np.pi * p), ra * (p - 0.5)
        elif pattern == "swing_roll":
            yaw, pitch, roll = ya * 0.5 * np.sin(2 * np.pi * p), pa * 0.5 * np.sin(3 * np.pi * p), ra * (p - 0.5)
        elif pattern == "pitch_roll":
            yaw, pitch, roll = ya * 0.5 * np.sin(2 * np.pi * p), pa * (p - 0.5), ra * 0.5 * np.sin(2 * np.pi * p)
        elif pattern == "double_whip":
            first = whip(np.clip(p * 2.0, 0.0, 1.0))
            second = whip(np.clip((p - 0.5) * 2.0, 0.0, 1.0))
            yaw, pitch, roll = ya * 0.5 * (first - second), pa * 0.5 * np.sin(2 * np.pi * p), np.zeros_like(p)
        else:
            raise ValueError(pattern)
        yaw, pitch, roll = yaw.astype(np.float32), pitch.astype(np.float32), roll.astype(np.float32)
    payload = np.stack([yaw, pitch, roll], axis=1).astype("<f4", copy=False).tobytes()
    return {
        "delta_yaw_deg": yaw,
        "delta_pitch_deg": pitch,
        "delta_roll_deg": roll,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def off_axis_geometry(width: int, height: int, case: dict[str, Any]) -> dict[str, Any]:
    """计算共同母平面和两个纵向偏轴裁框的内参位置。"""

    focal = width / (2.0 * math.tan(math.radians(OUTPUT_HFOV_DEG) / 2.0))
    layout = case["off_axis_layout"]
    gap = height * float(layout.get("crop_gap_ratio", CROP_GAP_RATIO))
    principal_y = height * float(
        layout.get("master_principal_y_ratio", MASTER_PRINCIPAL_Y_RATIO)
    )
    upper_origin = height * float(layout.get("upper_origin_y_ratio", 0.0))
    lower_origin = upper_origin + height + gap
    upper_side = str(layout.get("upper_side", "ref"))
    origins = {
        upper_side: upper_origin,
        "target" if upper_side == "ref" else "ref": lower_origin,
    }
    rows: dict[str, Any] = {}
    for side, origin_y in origins.items():
        top_angle = math.degrees(math.atan2(origin_y + 0.5 - principal_y, focal))
        bottom_angle = math.degrees(
            math.atan2(origin_y + height - 0.5 - principal_y, focal)
        )
        center_angle = math.degrees(
            math.atan2(origin_y + height / 2.0 - principal_y, focal)
        )
        rows[side] = {
            "origin_y": origin_y,
            "crop_role": "upper" if side == upper_side else "lower",
            "top_down_angle_deg": top_angle,
            "bottom_down_angle_deg": bottom_angle,
            "effective_center_pitch_deg": -center_angle,
        }
    upper_row = rows[upper_side]
    lower_side = "target" if upper_side == "ref" else "ref"
    lower_row = rows[lower_side]
    margin = lower_row["top_down_angle_deg"] - upper_row["bottom_down_angle_deg"]
    return {
        "focal_px": focal,
        "principal_x": width / 2.0,
        "principal_y": principal_y,
        "gap_px": gap,
        "upper_side": upper_side,
        "sides": rows,
        "vertical_nonoverlap_margin_deg": margin,
    }


def off_axis_rays(width: int, height: int, side: str, case: dict[str, Any]) -> np.ndarray:
    """生成共享外参、不同纵向主点位置的偏轴透视射线。"""

    geometry = off_axis_geometry(width, height, case)
    origin_y = float(geometry["sides"][side]["origin_y"])
    xs = (
        np.arange(width, dtype=np.float32)
        + 0.5
        - float(geometry["principal_x"])
    ) / float(geometry["focal_px"])
    ys = (
        np.arange(height, dtype=np.float32)
        + 0.5
        + origin_y
        - float(geometry["principal_y"])
    ) / float(geometry["focal_px"])
    xx, yy = np.meshgrid(xs, ys)
    rays = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size, np.float32)], axis=0)
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    return rays


def sampled_off_axis_longitudes(
    width: int,
    height: int,
    side: str,
    case: dict[str, Any],
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    geometry = off_axis_geometry(width, height, case)
    origin_y = float(geometry["sides"][side]["origin_y"])
    xs = np.linspace(0.5, width - 0.5, 5, dtype=np.float32)
    ys = np.linspace(0.5, height - 0.5, 5, dtype=np.float32)
    xx, yy = np.meshgrid(
        (xs - float(geometry["principal_x"])) / float(geometry["focal_px"]),
        (ys + origin_y - float(geometry["principal_y"])) / float(geometry["focal_px"]),
    )
    rays = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size, np.float32)], axis=0)
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    rotated = rotation_matrix(yaw_deg, pitch_deg, roll_deg) @ rays
    return np.degrees(np.arctan2(rotated[0], rotated[2]))


def trajectory_report(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    width, height = map(int, case["resolution"])
    geometry = off_axis_geometry(width, height, case)
    common = common_rotation(case)
    max_longitudes: dict[str, float] = {}
    for side in ("ref", "target"):
        base = case[f"{side}_view"]
        maximum = 0.0
        for index in list(range(0, FRAME_COUNT, 4)) + [FRAME_COUNT - 1]:
            longitude = sampled_off_axis_longitudes(
                width,
                height,
                side,
                case,
                float(base["yaw_deg"]) + float(common["delta_yaw_deg"][index]),
                float(base["pitch_deg"]) + float(common["delta_pitch_deg"][index]),
                float(base["roll_deg"]) + float(common["delta_roll_deg"][index]),
            )
            maximum = max(maximum, float(np.max(np.abs(longitude))))
        max_longitudes[side] = maximum
    return {
        "master_hfov_deg": OUTPUT_HFOV_DEG,
        "master_focal_px": round(float(geometry["focal_px"]), 6),
        "master_principal_y_px": round(float(geometry["principal_y"]), 6),
        "crop_gap_px": round(float(geometry["gap_px"]), 6),
        "upper_side": geometry["upper_side"],
        "crop_role": {
            side: geometry["sides"][side]["crop_role"] for side in ("ref", "target")
        },
        "crop_origin_y_px": {
            side: round(float(geometry["sides"][side]["origin_y"]), 6)
            for side in ("ref", "target")
        },
        "effective_center_pitch_deg": {
            side: round(float(geometry["sides"][side]["effective_center_pitch_deg"]), 6)
            for side in ("ref", "target")
        },
        "vertical_nonoverlap_margin_deg": round(float(geometry["vertical_nonoverlap_margin_deg"]), 6),
        "same_yaw": float(case["ref_view"]["yaw_deg"]) == float(case["target_view"]["yaw_deg"]),
        "same_master_extrinsics": case["ref_view"] == case["target_view"],
        "max_abs_sampled_longitude_deg": {key: round(value, 6) for key, value in max_longitudes.items()},
        "seam_safe": max(max_longitudes.values()) <= 162.0,
        "same_relative_rotation_sha256": common["sha256"],
        "source_speed_factor": float(case["source_speed_factor"]),
        "rotation_speed_multiplier": (
            float(case["virtual_rotation"]["speed_multiplier"])
            if case["virtual_rotation"]
            else 0.0
        ),
        "estimated_horizontal_source_pixels": round(3840.0 * OUTPUT_HFOV_DEG / 360.0, 3),
        "estimated_horizontal_upscale_ratio": round(
            width / (3840.0 * OUTPUT_HFOV_DEG / 360.0), 3
        ),
    }


def make_poster(video_path: Path, poster_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(video_path)
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(poster_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(poster_path)


def render_case(
    case: dict[str, Any],
    source_dir: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    case_dir = MEDIA_ROOT / case["id"]
    outputs = {side: case_dir / f"{side}.mp4" for side in ("ref", "target")}
    metadata_path = case_dir / "metadata.json"
    if not overwrite and metadata_path.exists() and all(path.exists() for path in outputs.values()):
        return {"id": case["id"], "status": "skipped_existing", "bytes": 0}

    source_path = source_dir / case["source"]["file"]
    reader = SharedClipReader(
        source_path,
        float(case["start_seconds"]),
        float(case["source_speed_factor"]),
    )
    width, height = map(int, case["resolution"])
    report = trajectory_report(case, config)
    if report["vertical_nonoverlap_margin_deg"] <= 3.0:
        raise RuntimeError(f"{case['id']} 上下取景框间隔不足：{report}")
    if not report["seam_safe"]:
        raise RuntimeError(f"{case['id']} 触及 ERP 接缝：{report}")
    common = common_rotation(case)
    rays_by_side = {
        side: off_axis_rays(width, height, side, case) for side in ("ref", "target")
    }
    writers = {
        side: RawFFmpegWriter(path, width, height, FPS, config)
        for side, path in outputs.items()
    }
    fixed_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    range_stats = {
        side: {
            "full": {"x_min": float("inf"), "x_max": float("-inf"), "y_min": float("inf"), "y_max": float("-inf")},
            "initial": None,
        }
        for side in ("ref", "target")
    }
    try:
        for frame_index in range(FRAME_COUNT):
            source_frame = reader.frame(frame_index)
            for side in ("ref", "target"):
                if case["kind"] == "translation_only" and side in fixed_maps:
                    maps = fixed_maps[side]
                else:
                    base = case[f"{side}_view"]
                    maps = rectilinear_maps(
                        rays_by_side[side],
                        (width, height),
                        (reader.width, reader.height),
                        float(base["yaw_deg"]) + float(common["delta_yaw_deg"][frame_index]),
                        float(base["pitch_deg"]) + float(common["delta_pitch_deg"][frame_index]),
                        float(base["roll_deg"]) + float(common["delta_roll_deg"][frame_index]),
                    )
                    if case["kind"] == "translation_only":
                        fixed_maps[side] = maps
                current_range = {
                    "x_min": float(np.min(maps[0])),
                    "x_max": float(np.max(maps[0])),
                    "y_min": float(np.min(maps[1])),
                    "y_max": float(np.max(maps[1])),
                }
                if frame_index == 0:
                    range_stats[side]["initial"] = dict(current_range)
                for key, value in current_range.items():
                    if key.endswith("min"):
                        range_stats[side]["full"][key] = min(
                            range_stats[side]["full"][key], value
                        )
                    else:
                        range_stats[side]["full"][key] = max(
                            range_stats[side]["full"][key], value
                        )
                output = cv2.remap(
                    source_frame,
                    maps[0],
                    maps[1],
                    cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                writers[side].write(output)
        for writer in writers.values():
            writer.close(True)
    except Exception:
        for writer in writers.values():
            try:
                writer.close(False)
            except Exception:
                pass
        for path in outputs.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        reader.close()

    for side, path in outputs.items():
        make_poster(path, case_dir / f"{side}.jpg")

    def public_range(values: dict[str, float]) -> dict[str, Any]:
        return {
            "erp_x_longitude_pixels": [
                int(math.floor(values["x_min"])),
                int(math.ceil(values["x_max"])),
            ],
            "longitude_degrees": [
                round((values["x_min"] / reader.width - 0.5) * 360.0, 3),
                round((values["x_max"] / reader.width - 0.5) * 360.0, 3),
            ],
            "erp_y_latitude_pixels": [
                int(math.floor(values["y_min"])),
                int(math.ceil(values["y_max"])),
            ],
            "latitude_degrees": [
                round(90.0 - values["y_max"] / reader.height * 180.0, 3),
                round(90.0 - values["y_min"] / reader.height * 180.0, 3),
            ],
        }

    erp_ranges = {
        side: {
            "crop_role": off_axis_geometry(width, height, case)["sides"][side]["crop_role"],
            "initial_frame": public_range(range_stats[side]["initial"]),
            "full_trajectory_union": public_range(range_stats[side]["full"]),
        }
        for side in ("ref", "target")
    }
    metadata = {
        "case": case,
        "trajectory_report": report,
        "erp_ranges": erp_ranges,
        "synchronization": {
            "single_shared_source_reader": True,
            "same_source_start_seconds": float(case["start_seconds"]),
            "same_source_frame_sequence": True,
            "same_yaw_deg": float(case["shared_yaw_deg"]),
            "ref_rotation_sha256": common["sha256"],
            "target_rotation_sha256": common["sha256"],
        },
        "rendered": {
            side: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for side, path in outputs.items()
        },
    }
    write_json(metadata_path, metadata)
    return {
        "id": case["id"],
        "status": "rendered",
        "bytes": sum(path.stat().st_size for path in outputs.values()),
    }


def render_all(plan: dict[str, Any], source_dir: Path, workers: int, overwrite: bool) -> None:
    config = load_render_config()
    payloads = [(case, source_dir, config, overwrite) for case in plan["cases"]]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for index, payload in enumerate(payloads, 1):
            result = render_case(*payload)
            results.append(result)
            print(f"[{index}/{len(payloads)}] {result['id']}: {result['status']}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(render_case, *payload) for payload in payloads]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(payloads)}] {result['id']}: {result['status']}", flush=True)
    print(f"完成 {len(results)} cases，约 {sum(x['bytes'] for x in results) / 1024**2:.1f} MiB")


def validate(plan: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    files: list[dict[str, Any]] = []
    if len(plan["cases"]) != 20:
        errors.append("case 数量不是 20")
    upper_target_count = sum(
        case["off_axis_layout"]["upper_side"] == "target" for case in plan["cases"]
    )
    if upper_target_count != 15:
        errors.append(f"上框分配给 TARGET 的数量不是 15：{upper_target_count}")
    rotation_speeds = [
        float(case["virtual_rotation"]["speed_multiplier"])
        for case in plan["cases"]
        if case["virtual_rotation"]
    ]
    if len(set(rotation_speeds)) < 8:
        errors.append("后十个旋转速度缺少随机差异")
    for case in plan["cases"]:
        if not (
            case["start_seconds"]
            == case["ref_start_seconds"]
            == case["target_start_seconds"]
        ):
            errors.append(f"{case['id']} 两侧时间不一致")
        if float(case["ref_view"]["yaw_deg"]) != float(case["target_view"]["yaw_deg"]):
            errors.append(f"{case['id']} 两侧经度不一致")
        if abs(float(case["source_speed_factor"]) - SOURCE_SPEED_FACTOR) > 1e-9:
            errors.append(f"{case['id']} 源移动速度不是 0.50×")
        metadata_path = MEDIA_ROOT / case["id"] / "metadata.json"
        if not metadata_path.exists():
            errors.append(f"缺少 {metadata_path.relative_to(PROJECT_ROOT)}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        report = metadata["trajectory_report"]
        sync = metadata["synchronization"]
        erp_ranges = metadata.get("erp_ranges", {})
        if not (
            report["same_yaw"]
            and report["same_master_extrinsics"]
            and report["vertical_nonoverlap_margin_deg"] > 3.0
            and report["seam_safe"]
            and sync["single_shared_source_reader"]
            and sync["same_source_frame_sequence"]
            and sync["ref_rotation_sha256"] == sync["target_rotation_sha256"]
        ):
            errors.append(f"{case['id']} 同经度/同步验证失败")
        if float(report.get("estimated_horizontal_upscale_ratio", 99.0)) > 2.0:
            errors.append(f"{case['id']} 有效源像素放大倍率仍过高")
        for side in ("ref", "target"):
            ranges = erp_ranges.get(side, {}).get("full_trajectory_union", {})
            x_px = ranges.get("erp_x_longitude_pixels", [])
            y_px = ranges.get("erp_y_latitude_pixels", [])
            if len(x_px) != 2 or len(y_px) != 2:
                errors.append(f"{case['id']} {side} 缺少 ERP 经纬度范围")
            elif not (0 <= x_px[0] <= x_px[1] <= 3840 and 0 <= y_px[0] <= y_px[1] <= 1920):
                errors.append(f"{case['id']} {side} ERP 范围越界")
            elif y_px[0] <= 2 or y_px[1] >= 1918:
                errors.append(f"{case['id']} {side} 过于接近 ERP 上下极点")
        pair_hashes: list[str] = []
        for side in ("ref", "target"):
            path = MEDIA_ROOT / case["id"] / f"{side}.mp4"
            meta = video_metadata(path)
            if not meta.get("decode_ok"):
                errors.append(f"无法解码 {path.relative_to(PROJECT_ROOT)}")
                continue
            width, height = map(int, case["resolution"])
            if (int(meta["width"]), int(meta["height"])) != (width, height):
                errors.append(f"{case['id']} {side} 分辨率错误")
            if width * height > MAX_PIXELS:
                errors.append(f"{case['id']} 分辨率超限")
            if abs(float(meta["fps"]) - FPS) > 0.05:
                errors.append(f"{case['id']} {side} fps 错误")
            if abs(float(meta["duration_seconds"]) - DURATION_SECONDS) > 0.06:
                errors.append(f"{case['id']} {side} 时长错误")
            digest = sha256_file(path)
            pair_hashes.append(digest)
            files.append(
                {
                    "case_id": case["id"],
                    "side": side,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "width": int(meta["width"]),
                    "height": int(meta["height"]),
                    "fps": round(float(meta["fps"]), 6),
                    "duration_seconds": round(float(meta["duration_seconds"]), 6),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        if len(pair_hashes) == 2 and pair_hashes[0] == pair_hashes[1]:
            errors.append(f"{case['id']} 两侧内容完全相同")
    result = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "case_count": len(plan["cases"]),
        "video_count": len(files),
        "counts": {
            "translation_only": sum(c["kind"] == "translation_only" for c in plan["cases"]),
            "translation_rotation": sum(
                c["kind"] == "translation_rotation" for c in plan["cases"]
            ),
        },
        "upper_window_assignment": {
            "target": upper_target_count,
            "ref": len(plan["cases"]) - upper_target_count,
        },
        "rotation_speed_multipliers": rotation_speeds,
        "rules": plan["rules"],
        "files": files,
    }
    write_json(VALIDATION_PATH, result)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise RuntimeError(f"验证失败：{len(errors)} 项")
    print("验证通过：20 cases / 40 MP4 / 同源同帧同经度")
    return result


def video_panel(case: dict[str, Any], side: str, metadata: dict[str, Any]) -> str:
    width, height = map(int, case["resolution"])
    geometry = off_axis_geometry(width, height, case)
    role = str(geometry["sides"][side]["crop_role"])
    role_zh = "上方" if role == "upper" else "下方"
    label = ("REF" if side == "ref" else "TARGET") + f" · {role_zh}取景框"
    pitch = float(geometry["sides"][side]["effective_center_pitch_deg"])
    ranges = metadata["erp_ranges"][side]["full_trajectory_union"]
    x_px = ranges["erp_x_longitude_pixels"]
    lon = ranges["longitude_degrees"]
    y_px = ranges["erp_y_latitude_pixels"]
    lat = ranges["latitude_degrees"]
    return f"""
      <figure class="video-panel">
        <figcaption><b>{label}</b><span>{escape(case['source']['file'])}</span></figcaption>
        <video controls playsinline preload="metadata" src="media/cases/{case['id']}/{side}.mp4" poster="media/cases/{case['id']}/{side}.jpg" style="aspect-ratio:{case['resolution'][0]}/{case['resolution'][1]}"></video>
        <div class="source-line"><b>同一源起点 {float(case['start_seconds']):.2f}s · yaw {float(case['shared_yaw_deg']):.0f}° · 有效中心 pitch {pitch:.1f}°</b>
        <span>全程经度（ERP 横向 x）{x_px[0]}–{x_px[1]} px · {lon[0]:.1f}°–{lon[1]:.1f}°</span>
        <span>全程纬度（ERP 纵向 y）{y_px[0]}–{y_px[1]} px · {lat[0]:.1f}°–{lat[1]:.1f}°</span></div>
      </figure>"""


def case_card(case: dict[str, Any]) -> str:
    title = "同经度上下双视角 · 原生移动"
    detail = "不叠加额外旋转"
    css_class = "pure"
    if case["kind"] == "translation_rotation":
        title = f"同经度上下双视角 · {escape(case['virtual_rotation']['name'])}"
        detail = (
            "两侧共享完全相同的逐帧 yaw / pitch / roll 增量 · "
            f"旋转速度倍率 {float(case['virtual_rotation']['speed_multiplier']):.2f}×"
        )
        css_class = "combined"
    width, height = case["resolution"]
    metadata = json.loads(
        (MEDIA_ROOT / case["id"] / "metadata.json").read_text(encoding="utf-8")
    )
    upper_side = str(case["off_axis_layout"]["upper_side"]).upper()
    return f"""
    <article class="case-card {css_class}" id="{case['id']}">
      <header><div><span class="case-id">{case['id']}</span><h3>{title}</h3></div>
      <div class="badges"><b>10.00s</b><b>{width}×{height}</b><b>源移动 0.50×</b><b>上框={upper_side}</b><b>同 yaw {float(case['shared_yaw_deg']):.0f}°</b></div></header>
      <p class="actions">随机纵向裁框 · 共享同一透视母平面外参 · 上下窗口互不重叠 · {detail}</p>
      <div class="video-pair" data-sync-pair>{video_panel(case, 'ref', metadata)}{video_panel(case, 'target', metadata)}</div>
    </article>"""


def build_site(plan: dict[str, Any]) -> None:
    pure = [case for case in plan["cases"] if case["kind"] == "translation_only"]
    combined = [case for case in plan["cases"] if case["kind"] == "translation_rotation"]
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同经度上下双视角 Cross-Pair</title>
<style>
:root{{--ink:#17211f;--paper:#f4f1e9;--card:#fffefb;--line:#d8d5ca;--blue:#2b5f8d;--green:#1d684f}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1320px;margin:auto;padding:22px 24px 70px}}
.hero,.diagram{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px;margin-bottom:18px}}h1{{font-size:clamp(25px,3vw,39px);line-height:1.2;margin:0}}h2,h3,p{{margin-top:0}}.diagram svg{{display:block;width:100%;height:auto}}.note{{font-weight:650;margin:10px 0 0}}
.section-title{{font-size:28px;margin:30px 0 12px}}.case-card{{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--blue);border-radius:18px;padding:18px;margin:16px 0}}.case-card.combined{{border-left-color:var(--green)}}.case-card>header{{display:flex;justify-content:space-between;gap:16px}}.case-id{{font:700 12px ui-monospace,monospace;color:#65716d}}.case-card h3{{margin:2px 0 5px;font-size:21px}}.badges{{display:flex;gap:7px;flex-wrap:wrap}}.badges b{{font-size:12px;background:#e7ece8;border-radius:999px;padding:5px 9px}}.actions{{font-weight:650}}
.video-pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}.video-panel{{margin:0;background:#101312;border-radius:13px;overflow:hidden}}.video-panel figcaption{{display:flex;justify-content:space-between;color:white;padding:9px 11px}}.video-panel video{{display:block;width:100%;background:#000;object-fit:contain}}.source-line{{color:#d2d9d5;padding:8px 11px;font-size:12px}}.source-line b,.source-line span{{display:block;margin:2px 0}}
@media(max-width:800px){{main{{padding:12px 10px 50px}}.case-card{{overflow-x:auto;padding:10px}}.video-pair{{min-width:720px}}.case-card>header{{display:block}}}}
</style></head><body><main>
<section class="hero"><h1>20 个 10 秒同经度上下双视角 Cross-Pair</h1></section>
<section class="diagram"><h2>固定同一 ERP 经度，只改变上下取景位置</h2>
<svg viewBox="0 0 1100 430" role="img" aria-label="ERP 同经度上下双取景框示意图">
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#1d684f"/></marker></defs>
<rect x="80" y="60" width="760" height="320" rx="18" fill="#e9f0f5" stroke="#2b5f8d" stroke-width="4"/>
<line x1="80" y1="60" x2="80" y2="380" stroke="#a44e2e" stroke-width="5" stroke-dasharray="10 8"/><line x1="840" y1="60" x2="840" y2="380" stroke="#a44e2e" stroke-width="5" stroke-dasharray="10 8"/>
<line x1="80" y1="220" x2="840" y2="220" stroke="#8a9691" stroke-width="3"/><text x="92" y="48" font-size="18">天空 / ERP 顶部</text><text x="92" y="410" font-size="18">地面 / ERP 底部</text><text x="350" y="245" font-size="17">赤道</text>
<rect x="390" y="225" width="170" height="60" rx="8" fill="#d7e6f2" stroke="#2b5f8d" stroke-width="5"/><rect x="390" y="305" width="170" height="60" rx="8" fill="#dcece4" stroke="#1d684f" stroke-width="5"/>
<line x1="475" y1="180" x2="475" y2="375" stroke="#17211f" stroke-width="3" stroke-dasharray="8 7"/><circle cx="475" cy="190" r="7" fill="#a44e2e"/><text x="570" y="262" font-size="20">上方偏轴裁框 · 角色随机</text><text x="570" y="342" font-size="20">下方偏轴裁框 · 角色随机</text><text x="870" y="120" font-size="18">共同母平面主点</text><text x="870" y="150" font-size="18">两个裁框都在主点下方</text>
<line x1="430" y1="278" x2="430" y2="245" stroke="#1d684f" stroke-width="6" marker-end="url(#arrow)"/><line x1="430" y1="358" x2="430" y2="325" stroke="#1d684f" stroke-width="6" marker-end="url(#arrow)"/><line x1="470" y1="262" x2="515" y2="262" stroke="#1d684f" stroke-width="6" marker-end="url(#arrow)"/><line x1="470" y1="342" x2="515" y2="342" stroke="#1d684f" stroke-width="6" marker-end="url(#arrow)"/>
</svg><p class="note">同一个 yaw、同一个源时间、同一张 ERP 帧、同一个透视母平面外参；每个 case 随机上下位置和 REF/TARGET 角色（上框偏向 TARGET），源移动统一降为 0.50×。页面不做镜像、稳定化或路径转移。</p></section>
<h2 class="section-title">10 个原生移动相机案例</h2>{''.join(case_card(case) for case in pure)}
<h2 class="section-title">10 个原生移动 + 同步旋转案例</h2>{''.join(case_card(case) for case in combined)}
</main><script>
document.querySelectorAll('[data-sync-pair]').forEach(pair=>{{const videos=[...pair.querySelectorAll('video')];let syncing=false,leader=videos[0],raf=null;const mirror=(source,eventName)=>{{if(syncing)return;syncing=true;videos.filter(v=>v!==source).forEach(v=>{{if(Math.abs(v.currentTime-source.currentTime)>.05)v.currentTime=source.currentTime;v.playbackRate=source.playbackRate;if(eventName==='play')v.play().catch(()=>{{}});if(eventName==='pause')v.pause();}});requestAnimationFrame(()=>syncing=false);}};const align=()=>{{if(!leader||leader.paused){{raf=null;return;}}syncing=true;videos.filter(v=>v!==leader).forEach(v=>{{if(Math.abs(v.currentTime-leader.currentTime)>.05)v.currentTime=leader.currentTime;v.playbackRate=leader.playbackRate;}});syncing=false;raf=requestAnimationFrame(align);}};videos.forEach(video=>{{['pointerdown','keydown'].forEach(name=>video.addEventListener(name,()=>leader=video));['play','pause','seeking','ratechange'].forEach(name=>video.addEventListener(name,()=>{{if(syncing)return;mirror(video,name);if(name==='play'&&raf===null)raf=requestAnimationFrame(align);}}));}});}});
</script></body></html>"""
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    INDEX_PATH.write_text(html, encoding="utf-8")
    print(f"网页已生成：{INDEX_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="同经度上下双视角 Cross-Pair")
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
        selected = dict(plan)
        selected["cases"] = plan["cases"][: args.limit] if args.limit else plan["cases"]
        render_all(selected, source_dir, max(1, args.workers), args.overwrite)
    if args.command in {"validate", "all"}:
        validate(plan)
    if args.command in {"site", "all"}:
        build_site(plan)


if __name__ == "__main__":
    main()
