#!/usr/bin/env python3
"""Panocam Motion V2 的唯一生产主程序。

这个文件故意把“素材筛选 → 生成计划 → 渲染 → 验证”放在同一个 CLI 中，
避免为了一个小改动在许多脚本之间来回跳。真正需要经常调整的数字全部放在
``config/pipeline.yaml``；本文件保留的是几何、质量控制和命令行流程。

四个子命令：

``audit``
    逐素材、逐 15 秒窗口计算动态分数和全局相机运动分数。静止画面、低帧率、
    非高清、时长不足、运动相机都会被拒绝。
``plan``
    固定随机种子生成 45 个动作配方；每个配方同时生成 sequential 与
    simultaneous 两个 case，再加入 5 个 tiny_planet 和 5 个 rabbit_hole。
``render``
    使用完全相同的逐帧轨迹渲染 ref/target。这里只做球面旋转与小幅 FOV 变化，
    没有深度、平移、视差或任何代理几何。
``validate``
    检查 100 个 case、200 个 MP4、15 秒、分辨率上限、左右一致性、动作白名单、
    顺序/叠加配对关系与接缝安全报告。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import cv2
import imageio_ffmpeg
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# 路径与白名单
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"
AUDIT_PATH = PROJECT_ROOT / "output" / "SOURCE_AUDIT.json"
PLAN_PATH = PROJECT_ROOT / "output" / "CASE_PLAN.json"
VALIDATION_PATH = PROJECT_ROOT / "output" / "VALIDATION.json"

# 组合运镜只允许以下 8 种 A 类动作。这里没有 static，因为用户要求不再生产单独
# A 类，组合中的“停顿”只是时间曲线的一部分，不应被伪装成一种动作。
ALLOWED_ACTIONS = (
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "roll_cw",
    "roll_ccw",
    "zoom_in",
    "zoom_out",
)

# 出现在计划里就应立即失败的旧版概念。验证阶段还会递归扫描全部字符串，防止
# 将代理平移或 C 类畸变误带回新版。
FORBIDDEN_TOKENS = (
    "proxy",
    "dolly",
    "truck",
    "pedestal",
    "crane",
    "arc_",
    "orbit",
    "distortion",
    "fisheye",
    "anamorphic",
    "focal_length",
    "B_transl",
    "C_optics",
)


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML，并对最容易填错的硬约束做一次快速检查。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    project = config["project"]
    expected_total = (
        int(project["sequential_cases"])
        + int(project["simultaneous_cases"])
        + int(project["tiny_planet_cases"])
        + int(project["rabbit_hole_cases"])
    )
    if expected_total != 100:
        raise ValueError(f"配置必须恰好生成 100 个 case，当前为 {expected_total}")
    if float(project["duration_seconds"]) != 15.0:
        raise ValueError("V2 的每条视频必须严格为 15 秒")
    max_pixels = int(project["max_pixels"])
    for width, height in config["resolutions"]:
        if width % 2 or height % 2:
            raise ValueError(f"H.264 yuv420p 要求偶数尺寸：{width}x{height}")
        if width * height > max_pixels:
            raise ValueError(f"分辨率超过 max_pixels：{width}x{height}")
    return config


def write_json(path: Path, payload: Any) -> None:
    """统一的 JSON 写入函数：UTF-8、中文不转义、末尾换行。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict[str, Any]:
    """用 OpenCV 读取源视频/结果视频的基础属性。"""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"decode_ok": False}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "decode_ok": bool(fps > 0 and frame_count > 0 and width > 0 and height > 0),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps if fps > 0 else 0.0,
    }


def resolve_source_roots(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    """解析素材根目录。

    YAML 中的相对路径都以“新项目根目录”为基准；命令行覆盖项优先，便于把公开
    仓库克隆到其他机器后使用自己的数据盘。
    """

    roots: dict[str, Path] = {}
    for dataset, raw in config["source_roots"].items():
        override = getattr(args, f"source_root_{dataset.replace('-', '_')}", None)
        path = Path(override).expanduser() if override else Path(raw).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        roots[dataset] = path
    return roots


def source_path_from_record(record: dict[str, Any], config: dict[str, Any]) -> Path:
    """由 dataset + file 在本机解析原始素材，不把绝对路径写进公开清单。"""

    raw_root = Path(config["source_roots"][record["dataset"]]).expanduser()
    if not raw_root.is_absolute():
        raw_root = (PROJECT_ROOT / raw_root).resolve()
    return raw_root / record["file"]


def sanitized_case(case: dict[str, Any]) -> dict[str, Any]:
    """删除早期/兼容计划中可能存在的本机绝对路径字段。"""

    clean = json.loads(json.dumps(case, ensure_ascii=False))
    for side in ("ref", "target"):
        if isinstance(clean.get(side), dict):
            clean[side].pop("path", None)
    return clean


# ---------------------------------------------------------------------------
# 素材筛选
# ---------------------------------------------------------------------------

def evenly_spaced_windows(duration: float, window: float, maximum: int) -> list[float]:
    """在整段视频中均匀抽取最多 maximum 个 15 秒候选窗口。"""

    latest = duration - window
    if latest < 0:
        return []
    if latest <= 2.0:
        return [max(0.0, latest / 2.0)]
    # 前后各留 1 秒，减少片头/片尾黑场对指标的污染。
    return [float(x) for x in np.linspace(1.0, max(1.0, latest - 1.0), maximum)]


def read_analysis_frames(
    path: Path,
    start_seconds: float,
    window_seconds: float,
    sample_count: int,
    analysis_size: tuple[int, int],
) -> list[np.ndarray]:
    """从一个 15 秒窗口均匀抽帧并缩小成灰度 ERP。

    筛选只需要判断“有没有动态”和“是否全局移动”，不需要在 5K 原图上算光流。
    先降到 512×256 可以把一次筛选的内存和时间降低两个数量级，而且保持 ERP 的
    全局运动结构。
    """

    # 不能对 5K 视频的每个样本反复随机 seek：多数编解码器会从前一个关键帧重新
    # 解码，导致同一段内容被重复解码十几次。这里让 ffmpeg 只 seek 一次，然后在
    # 解码管线内部降采样到灰度 512×256，速度通常快一个数量级。
    width, height = analysis_size
    sampling_fps = sample_count / window_seconds
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_seconds:.6f}",
        "-t", f"{window_seconds:.6f}",
        "-i", str(path),
        "-vf", f"fps={sampling_fps:.8f},scale={width}:{height}:flags=area,format=gray",
        "-frames:v", str(sample_count),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        return []
    frame_bytes = width * height
    available = len(completed.stdout) // frame_bytes
    return [
        np.frombuffer(
            completed.stdout[index * frame_bytes : (index + 1) * frame_bytes], dtype=np.uint8
        ).reshape(height, width).copy()
        for index in range(min(sample_count, available))
    ]


def analyse_window(path: Path, start: float, screening: dict[str, Any]) -> dict[str, Any]:
    """计算一个窗口的动态分数与全局运动分数。

    ``dynamic_score``
        相邻抽样帧中，灰度变化超过阈值的像素比例中位数。太低代表静止或近似静止。
    ``global_flow_px``
        稠密光流“每帧中位位移”的时间中位数。主体局部移动会抬高上分位数，但固定
        机位通常不会让全图一半以上像素同向移动；因此中位数比均值更稳健。
    ``coherent_shift_px``
        相位相关估计的整体平移。它专门补充识别全景相机统一水平/垂直转动。

    算法筛选不是版权或语义判断，所以 YAML 还保留人工 pass/reject 字段；即便人工
    标记 pass，具体 15 秒窗口仍必须通过动态与全局运动阈值。
    """

    width = int(screening["analysis_width"])
    height = int(screening["analysis_height"])
    frames = read_analysis_frames(
        path,
        start,
        float(screening["audit_window_seconds"]),
        int(screening["sample_count_per_window"]),
        (width, height),
    )
    if len(frames) < max(8, int(screening["sample_count_per_window"]) - 3):
        return {"start_seconds": start, "status": "reject", "reason": "decode_failed"}

    dynamic_ratios: list[float] = []
    global_flow: list[float] = []
    coherent_shifts: list[float] = []
    difference_threshold = int(screening["pixel_difference_threshold"])

    for previous, current in zip(frames, frames[1:]):
        difference = cv2.absdiff(previous, current)
        dynamic_ratios.append(float(np.mean(difference > difference_threshold)))

        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,  # 金字塔缩放比例
            3,    # 金字塔层数
            21,   # 邻域窗口；较大窗口更关注全局运动
            3,    # 每层迭代次数
            5,    # 多项式邻域
            1.2,  # 多项式高斯 sigma
            0,
        )
        magnitude = np.linalg.norm(flow, axis=2)
        global_flow.append(float(np.median(magnitude)))

        shift, response = cv2.phaseCorrelate(np.float32(previous), np.float32(current))
        if response > 0.08:
            coherent_shifts.append(float(math.hypot(*shift)))

    result = {
        "start_seconds": round(float(start), 3),
        "dynamic_score": round(float(np.median(dynamic_ratios)), 6),
        "global_flow_px": round(float(np.median(global_flow)), 6),
        "coherent_shift_px": round(
            float(np.median(coherent_shifts)) if coherent_shifts else 0.0,
            6,
        ),
    }
    if result["dynamic_score"] < float(screening["dynamic_score_min"]):
        result.update(status="reject", reason="static_or_nearly_static")
    elif (
        result["global_flow_px"] > float(screening["global_flow_px_max"])
        or result["coherent_shift_px"] > float(screening["coherent_shift_px_max"])
    ):
        result.update(status="reject", reason="moving_camera_or_global_motion")
    else:
        result.update(status="pass", reason="dynamic_fixed_camera_window")
    return result


def command_audit(args: argparse.Namespace, config: dict[str, Any]) -> None:
    roots = resolve_source_roots(config, args)
    screening = config["screening"]
    rows: list[dict[str, Any]] = []

    for source in config["sources"]:
        dataset = source["dataset"]
        path = roots[dataset] / source["file"]
        row: dict[str, Any] = {
            **source,
            "locator": f"{dataset}/{source['file']}",
            "path_exists": path.exists(),
        }
        if not path.exists():
            row.update(final_status="reject", final_reason="missing_file", usable_windows=[])
            rows.append(row)
            print(f"[reject] {source['file']}: 文件不存在")
            continue

        metadata = video_metadata(path)
        row["metadata"] = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in metadata.items()
        }
        basic_reasons: list[str] = []
        if not metadata["decode_ok"]:
            basic_reasons.append("decode_failed")
        if metadata.get("duration_seconds", 0) < float(screening["min_duration_seconds"]):
            basic_reasons.append("duration_too_short")
        if metadata.get("width", 0) < int(screening["min_width"]):
            basic_reasons.append("source_width_too_small")
        if metadata.get("height", 0) < int(screening["min_height"]):
            basic_reasons.append("source_height_too_small")
        if metadata.get("fps", 0) < float(screening["min_fps"]):
            basic_reasons.append("source_fps_too_low")
        if source.get("manual_decision") == "reject":
            basic_reasons.append("manual_content_reject")

        if basic_reasons:
            row.update(
                final_status="reject",
                final_reason="+".join(basic_reasons),
                usable_windows=[],
                analysed_windows=[],
            )
            rows.append(row)
            print(f"[reject] {source['file']}: {row['final_reason']}")
            continue

        starts = evenly_spaced_windows(
            float(metadata["duration_seconds"]),
            float(screening["audit_window_seconds"]),
            int(screening["max_windows_per_source"]),
        )
        windows = [analyse_window(path, start, screening) for start in starts]
        usable = [window for window in windows if window["status"] == "pass"]
        row["analysed_windows"] = windows
        row["usable_windows"] = usable
        if usable:
            row.update(final_status="pass", final_reason="manual_and_metric_checks_passed")
        else:
            row.update(final_status="reject", final_reason="no_usable_15_second_window")
        rows.append(row)
        print(f"[{row['final_status']}] {source['file']}: 可用窗口 {len(usable)}/{len(windows)}")

    payload = {
        "schema_version": 2,
        "generated_at_unix": time.time(),
        "screening_config": screening,
        "source_roots": dict(config["source_roots"]),
        "source_roots_note": "公开台账只记录配置中的逻辑路径；命令行覆盖的本机绝对路径不落盘",
        "summary": {
            "source_count": len(rows),
            "pass_count": sum(row["final_status"] == "pass" for row in rows),
            "reject_count": sum(row["final_status"] == "reject" for row in rows),
            "rule": "only_dynamic_scenes_from_stationary_360_cameras",
        },
        "sources": rows,
    }
    write_json(AUDIT_PATH, payload)
    print(f"已写入 {AUDIT_PATH}")


# ---------------------------------------------------------------------------
# 动作计划与轨迹
# ---------------------------------------------------------------------------

def amplitude_for_action(action: str, rng: random.Random, trajectory: dict[str, Any]) -> float:
    if action.startswith("pan_"):
        low, high = trajectory["pan_amplitude_deg"]
    elif action.startswith("tilt_"):
        low, high = trajectory["tilt_amplitude_deg"]
    elif action.startswith("roll_"):
        low, high = trajectory["roll_amplitude_deg"]
    elif action.startswith("zoom_"):
        low, high = trajectory["zoom_amplitude_deg"]
    else:
        raise ValueError(f"未知动作：{action}")
    return round(rng.uniform(float(low), float(high)), 3)


def base_hfov_for_resolution(width: int, height: int, trajectory: dict[str, Any]) -> float:
    """给竖屏自动降低水平 FOV，使垂直 FOV 不超过配置上限。"""

    base = math.radians(float(trajectory["base_hfov_deg"]))
    max_vertical = math.radians(float(trajectory["max_vertical_fov_deg"]))
    by_vertical_limit = 2.0 * math.atan(math.tan(max_vertical / 2.0) * width / height)
    result = min(base, by_vertical_limit)
    return math.degrees(result)


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def action_curve(
    normalized_time: np.ndarray,
    index: int,
    action_count: int,
    mode: str,
    phase: float,
    overlap_ratio: float,
    speed_multiplier: float,
    sequential_motion_fraction: float,
) -> np.ndarray:
    """返回单个动作在整条 15 秒内的 0～1 时间曲线。

    顺序组合：把时间均分成 N 段；当前动作在自己的段内快速完成，之后保持结果。
    叠加组合：每个动作使用一个宽脉冲，开始/结束略有错开，但公共重叠区至少达到
    配置比例。脉冲会回到 0，因此同轴反向动作也不会永久相互抵消。
    """

    if mode == "sequential":
        start = index / action_count
        full_end = (index + 1) / action_count
        # speed_multiplier 越大、motion_fraction 越小，动作越早完成；动作完成后保持
        # 最终角度直到下一个动作接力，因此不会凭空回弹。
        end = start + (full_end - start) * sequential_motion_fraction / speed_multiplier
        local = (normalized_time - start) / max(end - start, 1e-6)
        return smoothstep(local)

    if mode != "simultaneous":
        raise ValueError(f"未知组合模式：{mode}")
    maximum_edge = max(0.0, (1.0 - overlap_ratio) / 2.0)
    start = maximum_edge * phase
    end = 1.0 - maximum_edge * (1.0 - phase)
    local = (normalized_time - start) / max(end - start, 1e-6)
    # 叠加模式以 0.5 为中心压缩时间轴。倍率越大，脉冲越窄、角速度越快；所有
    # 动作仍在镜头中段同时达到显著幅度。
    local = (local - 0.5) * speed_multiplier + 0.5
    inside = (local >= 0.0) & (local <= 1.0)
    pulse = np.zeros_like(normalized_time)
    # sin(pi*x) 的起止速度为 0，中心达到完整幅度，避免硬切或突然转向。
    pulse[inside] = np.sin(np.pi * local[inside])
    return pulse


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """构造 R = Ry @ Rx @ Rz。

    坐标约定为 x 向右、y 向下、z 向前；矩阵只改变观察方向，不包含平移项。
    """

    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return (ry @ rx @ rz).astype(np.float32)


def sampled_view_longitudes(
    width: int,
    height: int,
    hfov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """用 5×5 视场采样点估计当前虚拟视角触及的源 ERP 经度。"""

    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    xs = np.linspace(-width / 2.0, width / 2.0, 5, dtype=np.float32) / focal
    ys = np.linspace(-height / 2.0, height / 2.0, 5, dtype=np.float32) / focal
    xx, yy = np.meshgrid(xs, ys)
    rays = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size, np.float32)], axis=0)
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    rotated = rotation_matrix(yaw_deg, pitch_deg, roll_deg) @ rays
    return np.degrees(np.arctan2(rotated[0], rotated[2]))


def build_trajectory(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """把动作列表变成逐帧 yaw/pitch/roll/hfov，并自动缩放到安全范围。"""

    project = config["project"]
    settings = config["trajectory"]
    fps = int(project["fps"])
    frame_count = int(round(float(project["duration_seconds"]) * fps))
    width, height = case["resolution"]
    normalized_time = np.arange(frame_count, dtype=np.float32) / max(frame_count - 1, 1)
    base_hfov = base_hfov_for_resolution(width, height, settings)

    raw_yaw = np.zeros(frame_count, np.float32)
    raw_pitch = np.zeros(frame_count, np.float32)
    raw_roll = np.zeros(frame_count, np.float32)
    raw_zoom = np.zeros(frame_count, np.float32)

    actions = case.get("actions", [])
    overlap = float(settings["simultaneous_min_overlap_ratio"])
    for index, item in enumerate(actions):
        action = item["type"]
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"动作不在 V2 白名单：{action}")
        curve = action_curve(
            normalized_time,
            index,
            len(actions),
            case["mode"],
            float(item["phase"]),
            overlap,
            float(item["speed_multiplier"]),
            float(settings["sequential_motion_fraction"]),
        )
        amount = float(item["amplitude_deg"]) * curve
        if action == "pan_left":
            raw_yaw -= amount
        elif action == "pan_right":
            raw_yaw += amount
        elif action == "tilt_up":
            raw_pitch -= amount
        elif action == "tilt_down":
            raw_pitch += amount
        elif action == "roll_cw":
            raw_roll += amount
        elif action == "roll_ccw":
            raw_roll -= amount
        elif action == "zoom_in":
            raw_zoom -= amount
        elif action == "zoom_out":
            raw_zoom += amount

    # 每个轴分别缩放，而不是粗暴 clip。clip 会在极值处形成一段完全不动的平台，
    # 缩放则保留完整时间曲线和相对速度。
    def fit(values: np.ndarray, absolute_limit: float) -> tuple[np.ndarray, float]:
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        scale = min(1.0, absolute_limit / peak) if peak > 0 else 1.0
        return values * scale, scale

    yaw, yaw_scale = fit(raw_yaw, float(settings["hard_yaw_center_abs_deg"]))
    pitch, pitch_scale = fit(raw_pitch, float(settings["hard_pitch_abs_deg"]))
    roll, roll_scale = fit(raw_roll, float(settings["hard_roll_abs_deg"]))

    zoom_low = float(settings["hard_hfov_min_deg"]) - base_hfov
    zoom_high = float(settings["hard_hfov_max_deg"]) - base_hfov
    zoom_scale = 1.0
    if float(raw_zoom.min(initial=0.0)) < zoom_low:
        zoom_scale = min(zoom_scale, zoom_low / float(raw_zoom.min()))
    if float(raw_zoom.max(initial=0.0)) > zoom_high:
        zoom_scale = min(zoom_scale, zoom_high / float(raw_zoom.max()))
    zoom = raw_zoom * zoom_scale
    hfov = base_hfov + zoom

    # 真实视场采样检查：所有普通组合都必须避开 ERP 左右接缝。若 roll/pitch 让角点
    # 比中心公式预计得更远，就等比收小全部旋转幅度，最多尝试 16 次。
    seam_margin = float(settings["seam_margin_deg"])
    longitude_limit = 180.0 - seam_margin
    seam_scale = 1.0
    max_abs_longitude = 0.0
    if case["kind"] == "combo":
        for _ in range(16):
            max_abs_longitude = 0.0
            # 每 4 帧检查一次，另外强制含最后一帧；轨迹是连续光滑函数。
            check_indices = list(range(0, frame_count, 4)) + [frame_count - 1]
            for frame_index in check_indices:
                longitudes = sampled_view_longitudes(
                    width,
                    height,
                    float(hfov[frame_index]),
                    float(yaw[frame_index] * seam_scale),
                    float(pitch[frame_index] * seam_scale),
                    float(roll[frame_index] * seam_scale),
                )
                max_abs_longitude = max(max_abs_longitude, float(np.max(np.abs(longitudes))))
            if max_abs_longitude <= longitude_limit:
                break
            seam_scale *= 0.9
        else:
            raise RuntimeError(f"轨迹无法避开 ERP 接缝：{case['id']}")
        yaw *= seam_scale
        pitch *= seam_scale
        roll *= seam_scale

    return {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "hfov_deg": hfov,
        "report": {
            "frame_count": frame_count,
            "base_hfov_deg": round(base_hfov, 6),
            "yaw_range_deg": [round(float(yaw.min()), 6), round(float(yaw.max()), 6)],
            "pitch_range_deg": [round(float(pitch.min()), 6), round(float(pitch.max()), 6)],
            "roll_range_deg": [round(float(roll.min()), 6), round(float(roll.max()), 6)],
            "hfov_range_deg": [round(float(hfov.min()), 6), round(float(hfov.max()), 6)],
            "axis_scales": {
                "yaw": round(yaw_scale, 6),
                "pitch": round(pitch_scale, 6),
                "roll": round(roll_scale, 6),
                "zoom": round(zoom_scale, 6),
                "seam": round(seam_scale, 6),
            },
            "max_abs_sampled_longitude_deg": round(max_abs_longitude, 6),
            "seam_longitude_limit_deg": round(longitude_limit, 6),
            "seam_safe": case["kind"] != "combo" or max_abs_longitude <= longitude_limit,
        },
    }


def pick_window(rng: random.Random, source: dict[str, Any]) -> dict[str, Any]:
    windows = source["usable_windows"]
    if not windows:
        raise ValueError(f"素材没有可用窗口：{source['file']}")
    return rng.choice(windows)


def public_source_record(source: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": source["dataset"],
        "file": source["file"],
        "start_seconds": window["start_seconds"],
        "dynamic_score": window["dynamic_score"],
        "global_flow_px": window["global_flow_px"],
        "coherent_shift_px": window["coherent_shift_px"],
        "license": source.get("license", "CC BY-NC-SA 4.0" if source["dataset"] == "360x" else "见来源台账"),
        "source_url": source.get("source_url", "https://x360dataset.github.io/" if source["dataset"] == "360x" else ""),
    }


def command_plan(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError("请先运行 audit，生成 output/SOURCE_AUDIT.json")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    pool = [row for row in audit["sources"] if row["final_status"] == "pass"]
    if len(pool) < 2:
        raise RuntimeError("至少需要 2 个通过筛选的固定机位动态素材")

    rng = random.Random(int(config["project"]["random_seed"]))
    recipes: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    resolutions = config["resolutions"]

    # 45 个配方平均覆盖动作数 2、3、4、5、6：每种动作数恰好 9 个配方。
    for recipe_index in range(int(config["project"]["combo_recipe_count"])):
        action_count = 2 + (recipe_index % 5)
        action_types = rng.sample(list(ALLOWED_ACTIONS), action_count)
        actions = [
            {
                "type": action,
                "amplitude_deg": amplitude_for_action(action, rng, config["trajectory"]),
                "speed_multiplier": round(
                    rng.uniform(*map(float, config["trajectory"]["speed_multiplier_range"])), 3
                ),
                # phase 只影响叠加模式中宽脉冲的轻微错开；0/1 都仍保留公共重叠区。
                "phase": round(rng.random(), 4),
                "parameter_hint": "调 amplitude_deg 改幅度；调 speed_multiplier 改速度；调 phase 改叠加起止错位",
            }
            for action in action_types
        ]
        # 先按普通随机流程完整消耗随机数，再覆盖两个历史纯 A 配方。这样后续 recipe
        # 的源素材、窗口和参数不发生漂移；已经完成的其他渲染也仍与新计划一致。
        display_name = None
        origin = "deterministic_random_expansion"
        if recipe_index == 40:  # recipe_41，2 动作槽
            display_name = "Vertigo Roll · 旋转眩晕（旧版合法组合保留）"
            origin = "legacy_retained_exact_A_only"
            preset = [("roll_cw", 22.0), ("zoom_in", 9.0)]
            for item, (action_type, amplitude) in zip(actions, preset):
                item["type"] = action_type
                item["amplitude_deg"] = amplitude
        elif recipe_index == 41:  # recipe_42，3 动作槽
            display_name = "Pan → Zoom → Tilt（旧版 explicit 组合保留）"
            origin = "legacy_retained_exact_A_only"
            preset = [("pan_right", 62.0), ("zoom_in", 9.0), ("tilt_up", 34.0)]
            for item, (action_type, amplitude) in zip(actions, preset):
                item["type"] = action_type
                item["amplitude_deg"] = amplitude
        ref_source, target_source = rng.sample(pool, 2)
        ref_window = pick_window(rng, ref_source)
        target_window = pick_window(rng, target_source)
        recipe = {
            "id": f"recipe_{recipe_index + 1:02d}",
            "display_name": display_name,
            "origin": origin,
            "action_count": action_count,
            "actions": actions,
            "resolution": list(resolutions[recipe_index % len(resolutions)]),
            "ref": public_source_record(ref_source, ref_window),
            "target": public_source_record(target_source, target_window),
        }
        recipes.append(recipe)
        for mode in ("sequential", "simultaneous"):
            case = {
                "id": f"{recipe['id']}_{'seq' if mode == 'sequential' else 'sim'}",
                "kind": "combo",
                "mode": mode,
                "recipe_id": recipe["id"],
                "display_name": recipe["display_name"],
                "origin": recipe["origin"],
                "action_count": action_count,
                "actions": actions,
                "resolution": recipe["resolution"],
                "duration_seconds": 15.0,
                "fps": int(config["project"]["fps"]),
                "ref": recipe["ref"],
                "target": recipe["target"],
            }
            trajectory = build_trajectory(case, config)
            case["trajectory_report"] = trajectory["report"]
            cases.append(case)

    # 特殊投影不组合任何 A 类动作，因此 actions 为空、mode=standalone。
    for kind, count in (
        ("tiny_planet", int(config["project"]["tiny_planet_cases"])),
        ("rabbit_hole", int(config["project"]["rabbit_hole_cases"])),
    ):
        for index in range(count):
            ref_source, target_source = rng.sample(pool, 2)
            case = {
                "id": f"{kind}_{index + 1:02d}",
                "kind": kind,
                "mode": "standalone",
                "recipe_id": None,
                "action_count": 0,
                "actions": [],
                "resolution": list(resolutions[(90 + index + (0 if kind == 'tiny_planet' else 5)) % len(resolutions)]),
                "duration_seconds": 15.0,
                "fps": int(config["project"]["fps"]),
                "ref": public_source_record(ref_source, pick_window(rng, ref_source)),
                "target": public_source_record(target_source, pick_window(rng, target_source)),
                "projection_parameters": {
                    "stereographic_scale": round(0.92 + 0.04 * index, 3),
                    "seam_orientation_deg": float(-144 + 72 * index),
                    "note": "特殊投影需要使用完整 360°；接缝方向旋转到画面中相对不显眼的方位",
                },
                "trajectory_report": {
                    "frame_count": int(15 * config["project"]["fps"]),
                    "seam_safe": None,
                    "note": "全 360° 特殊投影无法像普通透视视角一样完全排除 ERP 接缝",
                },
            }
            cases.append(case)

    # 网页按 recipe 相邻展示 seq/sim；特殊投影放在最后。
    payload = {
        "schema_version": 2,
        "random_seed": int(config["project"]["random_seed"]),
        "summary": {
            "total_cases": len(cases),
            "combo_cases": sum(case["kind"] == "combo" for case in cases),
            "sequential_cases": sum(case["mode"] == "sequential" for case in cases),
            "simultaneous_cases": sum(case["mode"] == "simultaneous" for case in cases),
            "tiny_planet_cases": sum(case["kind"] == "tiny_planet" for case in cases),
            "rabbit_hole_cases": sum(case["kind"] == "rabbit_hole" for case in cases),
            "videos": len(cases) * 2,
            "duration_seconds_each": 15.0,
            "max_pixels": int(config["project"]["max_pixels"]),
        },
        "recipes": recipes,
        "cases": cases,
    }
    write_json(PLAN_PATH, payload)
    print(f"已生成 {len(cases)} 个 case：{PLAN_PATH}")


# ---------------------------------------------------------------------------
# 球面投影与视频渲染
# ---------------------------------------------------------------------------

def rectilinear_rays(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """生成一个普通透视虚拟相机的单位射线，形状为 3×(W*H)。"""

    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    xs = (np.arange(width, dtype=np.float32) + 0.5 - width / 2.0) / focal
    ys = (np.arange(height, dtype=np.float32) + 0.5 - height / 2.0) / focal
    xx, yy = np.meshgrid(xs, ys)
    rays = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size, np.float32)], axis=0)
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)
    return rays


def rectilinear_maps(
    rays: np.ndarray,
    output_size: tuple[int, int],
    source_size: tuple[int, int],
    yaw: float,
    pitch: float,
    roll: float,
) -> tuple[np.ndarray, np.ndarray]:
    """把虚拟相机射线旋转后转换为 ERP 采样坐标。

    map_x 使用模运算只为数值稳健；普通组合在此前已经通过接缝安全检查，正常情况
    不会真正跨越 x=0 / x=W 的拼接边界。
    """

    width, height = output_size
    source_width, source_height = source_size
    rotated = rotation_matrix(yaw, pitch, roll) @ rays
    longitude = np.arctan2(rotated[0], rotated[2])
    latitude_down = np.arcsin(np.clip(rotated[1], -1.0, 1.0))
    map_x = ((longitude / (2.0 * np.pi) + 0.5) * source_width) % source_width
    map_y = np.clip((latitude_down / np.pi + 0.5) * source_height, 0, source_height - 1)
    return map_x.reshape(height, width).astype(np.float32), map_y.reshape(height, width).astype(np.float32)


def stereographic_maps(
    output_size: tuple[int, int],
    source_size: tuple[int, int],
    kind: str,
    scale: float,
    orientation_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """生成小行星/兔子洞的固定球极投影映射。

    中心落在南极得到 tiny planet，落在北极得到 rabbit hole。它们都使用完整 360°
    经度，所以只能通过 orientation 调整接缝方向，不能像普通视角那样完全排除接缝。
    """

    width, height = output_size
    source_width, source_height = source_size
    side = float(min(width, height))
    xs = (np.arange(width, dtype=np.float32) + 0.5 - width / 2.0) / (side * 0.5)
    ys = (np.arange(height, dtype=np.float32) + 0.5 - height / 2.0) / (side * 0.5)
    xx, yy = np.meshgrid(xs, ys)
    radius = np.sqrt(xx * xx + yy * yy)
    theta = 2.0 * np.arctan2(radius, float(scale))
    longitude = np.arctan2(xx, -yy) + math.radians(orientation_deg)
    # 注意这里的 latitude 是“向下为正”的 ERP 纬角：+π/2 是地面/天底，-π/2
    # 是天空/天顶。若沿用地理纬度的正负号，小行星和兔子洞会被整体对调。
    if kind == "tiny_planet":
        latitude = np.pi / 2.0 - theta
    elif kind == "rabbit_hole":
        latitude = -np.pi / 2.0 + theta
    else:
        raise ValueError(f"未知特殊投影：{kind}")
    latitude = np.clip(latitude, -np.pi / 2.0, np.pi / 2.0)
    map_x = ((longitude / (2.0 * np.pi) + 0.5) * source_width) % source_width
    map_y = np.clip((latitude / np.pi + 0.5) * source_height, 0, source_height - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32)


class SequentialClipReader:
    """按目标 fps 从一个连续源窗口顺序读取，不对每一帧反复 seek。"""

    def __init__(self, path: Path, start_seconds: float, output_fps: int):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开源视频：{path}")
        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.source_fps <= 0:
            raise RuntimeError(f"源视频 fps 无效：{path}")
        self.output_fps = output_fps
        self.start_frame = int(round(start_seconds * self.source_fps))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.current_source_index = self.start_frame - 1
        self.current_frame: np.ndarray | None = None
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frame_for_output(self, output_index: int) -> np.ndarray:
        desired = self.start_frame + int(round(output_index * self.source_fps / self.output_fps))
        while self.current_source_index < desired:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"源视频在 15 秒窗口结束前解码失败：{self.path}")
            self.current_source_index += 1
            self.current_frame = frame
        assert self.current_frame is not None
        return self.current_frame

    def close(self) -> None:
        self.capture.release()


class RawFFmpegWriter:
    """把 OpenCV BGR 帧直接送给 ffmpeg，编码完成后原子替换临时文件。"""

    def __init__(self, final_path: Path, width: int, height: int, fps: int, config: dict[str, Any]):
        final_path.parent.mkdir(parents=True, exist_ok=True)
        self.final_path = final_path
        self.temp_path = final_path.with_name(final_path.stem + ".part.mp4")
        self.temp_path.unlink(missing_ok=True)
        render = config["render"]
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s:v", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", str(render["codec"]),
            "-preset", str(render["preset"]),
            "-crf", str(render["crf"]),
            "-maxrate", str(render["maxrate"]),
            "-bufsize", str(render["bufsize"]),
            "-pix_fmt", str(render["pixel_format"]),
            "-threads", str(render["ffmpeg_threads"]),
            "-movflags", "+faststart",
            str(self.temp_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin 不可用")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self, success: bool = True) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if not success or return_code != 0:
            self.temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg 编码失败：{self.final_path}，返回码 {return_code}")
        os.replace(self.temp_path, self.final_path)


def render_case(case: dict[str, Any], config: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    """渲染一个 Cross-Pair case；两路共享同一张逐帧采样表。"""

    case_dir = PROJECT_ROOT / "site" / "media" / "cases" / case["id"]
    ref_output = case_dir / "ref.mp4"
    target_output = case_dir / "target.mp4"
    metadata_output = case_dir / "metadata.json"
    if not overwrite and ref_output.exists() and target_output.exists() and metadata_output.exists():
        return {"id": case["id"], "status": "skipped_existing"}

    width, height = map(int, case["resolution"])
    fps = int(case["fps"])
    frame_count = int(round(float(case["duration_seconds"]) * fps))
    ref_reader = SequentialClipReader(
        source_path_from_record(case["ref"], config), float(case["ref"]["start_seconds"]), fps
    )
    target_reader = SequentialClipReader(
        source_path_from_record(case["target"], config), float(case["target"]["start_seconds"]), fps
    )
    ref_writer = RawFFmpegWriter(ref_output, width, height, fps, config)
    target_writer = RawFFmpegWriter(target_output, width, height, fps, config)

    trajectory: dict[str, Any] | None = None
    try:
        if case["kind"] == "combo":
            trajectory = build_trajectory(case, config)
        else:
            params = case["projection_parameters"]
            ref_special_maps = stereographic_maps(
                (width, height),
                (ref_reader.width, ref_reader.height),
                case["kind"],
                float(params["stereographic_scale"]),
                float(params["seam_orientation_deg"]),
            )
            target_special_maps = stereographic_maps(
                (width, height),
                (target_reader.width, target_reader.height),
                case["kind"],
                float(params["stereographic_scale"]),
                float(params["seam_orientation_deg"]),
            )

        # 当 FOV 变化时射线也变化。缓存按 0.01° 四舍五入后的射线，可减少平缓 zoom
        # 曲线中的重复三角计算；没有 zoom 的 case 通常只需要一份缓存。
        ray_cache: dict[float, np.ndarray] = {}
        for frame_index in range(frame_count):
            ref_source = ref_reader.frame_for_output(frame_index)
            target_source = target_reader.frame_for_output(frame_index)
            if case["kind"] == "combo":
                assert trajectory is not None
                hfov = float(trajectory["hfov_deg"][frame_index])
                cache_key = round(hfov, 2)
                rays = ray_cache.get(cache_key)
                if rays is None:
                    rays = rectilinear_rays(width, height, cache_key)
                    ray_cache[cache_key] = rays
                common = (
                    float(trajectory["yaw_deg"][frame_index]),
                    float(trajectory["pitch_deg"][frame_index]),
                    float(trajectory["roll_deg"][frame_index]),
                )
                ref_maps = rectilinear_maps(
                    rays, (width, height), (ref_reader.width, ref_reader.height), *common
                )
                target_maps = rectilinear_maps(
                    rays, (width, height), (target_reader.width, target_reader.height), *common
                )
            else:
                ref_maps = ref_special_maps
                target_maps = target_special_maps

            ref_frame = cv2.remap(
                ref_source, ref_maps[0], ref_maps[1], cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
            )
            target_frame = cv2.remap(
                target_source,
                target_maps[0],
                target_maps[1],
                cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            ref_writer.write(ref_frame)
            target_writer.write(target_frame)

        ref_writer.close(True)
        target_writer.close(True)
    except Exception:
        # 无论哪一路失败，都不保留“看似完整”的另一半，避免网页出现半个 case。
        try:
            ref_writer.close(False)
        except Exception:
            pass
        try:
            target_writer.close(False)
        except Exception:
            pass
        ref_output.unlink(missing_ok=True)
        target_output.unlink(missing_ok=True)
        raise
    finally:
        ref_reader.close()
        target_reader.close()

    metadata = {
        "case": sanitized_case(case),
        "rendered": {
            "ref": str(ref_output.relative_to(PROJECT_ROOT)),
            "target": str(target_output.relative_to(PROJECT_ROOT)),
            "ref_bytes": ref_output.stat().st_size,
            "target_bytes": target_output.stat().st_size,
            "ref_sha256": sha256_file(ref_output),
            "target_sha256": sha256_file(target_output),
            "codec": config["render"],
        },
        "trajectory_report": trajectory["report"] if trajectory else case["trajectory_report"],
    }
    write_json(metadata_output, metadata)
    return {"id": case["id"], "status": "rendered", "bytes": ref_output.stat().st_size + target_output.stat().st_size}


def render_worker(payload: tuple[dict[str, Any], dict[str, Any], bool]) -> dict[str, Any]:
    case, config, overwrite = payload
    return render_case(case, config, overwrite)


def command_render(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not PLAN_PATH.exists():
        raise FileNotFoundError("请先运行 plan，生成 output/CASE_PLAN.json")
    # render 也允许覆盖素材根目录；覆盖值只进入当前进程，不写入公开 JSON。
    for dataset in config["source_roots"]:
        override = getattr(args, f"source_root_{dataset.replace('-', '_')}", None)
        if override:
            config["source_roots"][dataset] = str(Path(override).expanduser().resolve())
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    cases = plan["cases"]
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"CASE_PLAN 中不存在：{sorted(missing)}")
    if args.limit is not None:
        cases = cases[: int(args.limit)]
    workers = max(1, int(args.workers))
    payloads = [(case, config, bool(args.overwrite)) for case in cases]
    rendered_bytes = 0

    if workers == 1:
        for index, payload in enumerate(payloads, 1):
            result = render_worker(payload)
            rendered_bytes += int(result.get("bytes", 0))
            print(f"[{index}/{len(payloads)}] {result['id']}: {result['status']}", flush=True)
    else:
        # 每个 worker 自己再开两路 ffmpeg；普通笔记本建议 workers=2。继续加大可能
        # 因同时解码多路 5K 视频而变慢或耗尽内存。
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(render_worker, payload) for payload in payloads]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                rendered_bytes += int(result.get("bytes", 0))
                print(f"[{index}/{len(payloads)}] {result['id']}: {result['status']}", flush=True)
    print(f"本次完成 {len(payloads)} 个 case，新写入约 {rendered_bytes / 1024**2:.1f} MiB")


# ---------------------------------------------------------------------------
# 最终验证
# ---------------------------------------------------------------------------

def all_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from all_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_string_values(child)


def command_validate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not PLAN_PATH.exists():
        raise FileNotFoundError("缺少 output/CASE_PLAN.json")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    cases = plan["cases"]
    errors: list[str] = []
    file_rows: list[dict[str, Any]] = []
    max_pixels = int(config["project"]["max_pixels"])

    if len(cases) != 100:
        errors.append(f"case 数量不是 100：{len(cases)}")
    counts = {
        "combo": sum(case["kind"] == "combo" for case in cases),
        "sequential": sum(case["mode"] == "sequential" for case in cases),
        "simultaneous": sum(case["mode"] == "simultaneous" for case in cases),
        "tiny_planet": sum(case["kind"] == "tiny_planet" for case in cases),
        "rabbit_hole": sum(case["kind"] == "rabbit_hole" for case in cases),
    }
    expected_counts = {"combo": 90, "sequential": 45, "simultaneous": 45, "tiny_planet": 5, "rabbit_hole": 5}
    if counts != expected_counts:
        errors.append(f"类目计数不正确：{counts}")

    # 每个 recipe 必须同时拥有顺序与叠加，而且动作列表完全一致。
    recipe_modes: dict[str, dict[str, dict[str, Any]]] = {}
    for case in cases:
        if case["kind"] == "combo":
            recipe_modes.setdefault(case["recipe_id"], {})[case["mode"]] = case
            if not 2 <= int(case["action_count"]) <= 6:
                errors.append(f"{case['id']} 动作数不在 2～6")
            for item in case["actions"]:
                if item["type"] not in ALLOWED_ACTIONS:
                    errors.append(f"{case['id']} 含非法动作 {item['type']}")
            if not case["trajectory_report"].get("seam_safe"):
                errors.append(f"{case['id']} 接缝安全检查未通过")
        elif case["actions"] or case["action_count"] != 0:
            errors.append(f"{case['id']} 特殊投影不应组合动作")

    for recipe_id, modes in recipe_modes.items():
        if set(modes) != {"sequential", "simultaneous"}:
            errors.append(f"{recipe_id} 缺少顺序或叠加版本")
        elif modes["sequential"]["actions"] != modes["simultaneous"]["actions"]:
            errors.append(f"{recipe_id} 两种模式的动作配方不一致")

    combined_text = "\n".join(all_string_values(plan)).lower()
    for token in FORBIDDEN_TOKENS:
        if token.lower() in combined_text:
            errors.append(f"计划中出现禁用概念：{token}")

    for case in cases:
        expected_width, expected_height = map(int, case["resolution"])
        if expected_width * expected_height > max_pixels:
            errors.append(f"{case['id']} 计划分辨率超限")
        pair_rows = []
        for side in ("ref", "target"):
            path = PROJECT_ROOT / "site" / "media" / "cases" / case["id"] / f"{side}.mp4"
            if not path.exists():
                errors.append(f"缺少文件：{path.relative_to(PROJECT_ROOT)}")
                continue
            meta = video_metadata(path)
            duration = float(meta.get("duration_seconds", 0.0))
            if not meta.get("decode_ok"):
                errors.append(f"无法解码：{path.relative_to(PROJECT_ROOT)}")
            if (meta.get("width"), meta.get("height")) != (expected_width, expected_height):
                errors.append(f"{case['id']} {side} 分辨率不符：{meta}")
            if abs(float(meta.get("fps", 0)) - float(case["fps"])) > 0.05:
                errors.append(f"{case['id']} {side} fps 不符")
            if abs(duration - 15.0) > 0.06:
                errors.append(f"{case['id']} {side} 时长不是 15 秒：{duration}")
            row = {
                "case_id": case["id"],
                "side": side,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "pixels": int(meta.get("width", 0)) * int(meta.get("height", 0)),
                "fps": round(float(meta.get("fps", 0)), 6),
                "duration_seconds": round(duration, 6),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            # 网页有 200 个播放器，不能 preload=auto 把约 700 MiB 全部抢先下载。
            # 每路生成一张 640px 内的预览图：页面立即可见，点击播放后仍是原始高清 MP4。
            poster_path = path.with_suffix(".jpg")
            poster_capture = cv2.VideoCapture(str(path))
            poster_capture.set(cv2.CAP_PROP_POS_MSEC, 1000.0)
            poster_ok, poster = poster_capture.read()
            poster_capture.release()
            if poster_ok:
                scale = min(1.0, 640.0 / max(poster.shape[1], poster.shape[0]))
                if scale < 1.0:
                    poster = cv2.resize(
                        poster,
                        (int(round(poster.shape[1] * scale)), int(round(poster.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imwrite(str(poster_path), poster, [cv2.IMWRITE_JPEG_QUALITY, 88])
                row["poster"] = str(poster_path.relative_to(PROJECT_ROOT))
            file_rows.append(row)
            pair_rows.append(row)
        if len(pair_rows) == 2:
            if (pair_rows[0]["width"], pair_rows[0]["height"]) != (pair_rows[1]["width"], pair_rows[1]["height"]):
                errors.append(f"{case['id']} ref/target 分辨率不一致")
            if pair_rows[0]["sha256"] == pair_rows[1]["sha256"]:
                errors.append(f"{case['id']} ref/target 内容完全相同")
            # validation 同时重建一份可公开的逐 case 元数据；这样公开仓库不会泄露
            # /Users/... 等本机绝对路径，复现时按 config.source_roots 重新解析。
            metadata_path = PROJECT_ROOT / "site" / "media" / "cases" / case["id"] / "metadata.json"
            write_json(
                metadata_path,
                {
                    "case": sanitized_case(case),
                    "rendered": {
                        "ref": pair_rows[0],
                        "target": pair_rows[1],
                        "codec": config["render"],
                    },
                    "trajectory_report": case["trajectory_report"],
                },
            )

    validation = {
        "schema_version": 2,
        "passed": not errors,
        "errors": errors,
        "counts": counts,
        "expected_counts": expected_counts,
        "case_count": len(cases),
        "video_count": len(file_rows),
        "total_video_bytes": sum(row["bytes"] for row in file_rows),
        "max_pixels": max_pixels,
        "rules": {
            "duration_seconds_each": 15.0,
            "same_resolution_within_pair": True,
            "different_resolutions_across_cases": True,
            "only_A_rotation_zoom_combinations": True,
            "no_proxy_geometry": True,
            "normal_combo_seam_avoidance": True,
        },
        "files": file_rows,
    }
    write_json(VALIDATION_PATH, validation)
    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}", file=sys.stderr)
        raise RuntimeError(f"验证失败，共 {len(errors)} 项；详见 {VALIDATION_PATH}")
    print(f"验证通过：100 cases / 200 MP4 / 每条 15 秒。报告：{VALIDATION_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Panocam Motion V2 生产管线")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML 配置文件")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="筛选固定机位动态全景素材")
    audit.add_argument("--source-root-360x", help="覆盖 360+x 素材目录")
    audit.add_argument("--source-root-commons", help="覆盖 Wikimedia 素材目录")

    subparsers.add_parser("plan", help="生成 100-case 确定性计划")

    render = subparsers.add_parser("render", help="渲染 ref/target MP4")
    render.add_argument("--workers", type=int, default=1, help="并行 case 数；推荐 1～2")
    render.add_argument("--limit", type=int, help="只渲染计划开头 N 条，用于冒烟测试")
    render.add_argument("--case", action="append", help="只渲染指定 case id，可重复传入")
    render.add_argument("--overwrite", action="store_true", help="覆盖已经完整存在的 case")
    render.add_argument("--source-root-360x", help="本次渲染覆盖 360+x 素材目录")
    render.add_argument("--source-root-commons", help="本次渲染覆盖 Wikimedia 素材目录")

    subparsers.add_parser("validate", help="验证 100 cases 与 200 个视频")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    commands = {
        "audit": command_audit,
        "plan": command_plan,
        "render": command_render,
        "validate": command_validate,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
