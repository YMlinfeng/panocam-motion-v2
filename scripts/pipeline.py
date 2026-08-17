#!/usr/bin/env python3
"""Panocam Motion V2 的唯一生产主程序。

这个文件故意把“素材筛选 → 生成计划 → 渲染 → 验证”放在同一个 CLI 中，
避免为了一个小改动在许多脚本之间来回跳。真正需要经常调整的数字全部放在
``config/pipeline.yaml``；本文件保留的是几何、质量控制和命令行流程。

四个子命令：

``audit``
    逐素材、逐 15 秒窗口计算动态分数、切镜分数和全局相机运动分数。低帧率、
    非高清、时长不足、运动相机与剪辑转场都会被拒绝；只有显式标记的固定机位
    静止场景可以跳过最低动态分数。
``plan``
    固定随机种子生成 45 个动作配方；每个配方生成两个独立的错峰叠加混合时序
    case。所有 case 都有覆盖 15 秒的持续运动骨架，其他动作在不同时间窗叠加，
    因此不会在动作衔接处停顿。最后加入 5 个 tiny_planet 和 5 个 rabbit_hole。
``render``
    使用完全相同的逐帧轨迹渲染 ref/target。这里只做球面旋转与小幅 FOV 变化，
    没有深度、平移、视差或任何代理几何。
``validate``
    检查 100 个 case、200 个 MP4、15 秒、分辨率上限、左右一致性、动作白名单、
    连续运动、快速/大幅偏置、素材角色、低重复分配与接缝安全报告。
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

# 组合运镜只允许以下 8 种 A 类动作。这里没有 static；每个混合时序 case 都由
# 持续 backbone + 多个错峰时间窗构成，任何停顿都应被验证器当成错误。
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
    "proxy_orbit",
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
        int(project["hybrid_a_cases"])
        + int(project["hybrid_b_cases"])
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


def analyse_window(
    path: Path,
    start: float,
    screening: dict[str, Any],
    allow_static_scene: bool = False,
) -> dict[str, Any]:
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
        "dynamic_score_p95": round(float(np.percentile(dynamic_ratios, 95)), 6),
        "dynamic_score_max": round(float(np.max(dynamic_ratios)), 6),
        "global_flow_px": round(float(np.median(global_flow)), 6),
        "coherent_shift_px": round(
            float(np.median(coherent_shifts)) if coherent_shifts else 0.0,
            6,
        ),
    }
    if result["dynamic_score_max"] > float(screening["scene_cut_dynamic_ratio_max"]):
        result.update(status="reject", reason="scene_cut_or_full_frame_transition")
    elif (
        result["global_flow_px"] > float(screening["global_flow_px_max"])
        or result["coherent_shift_px"] > float(screening["coherent_shift_px_max"])
    ):
        result.update(status="reject", reason="moving_camera_or_global_motion")
    elif result["dynamic_score"] < float(screening["dynamic_score_min"]):
        if allow_static_scene:
            result.update(status="pass", reason="static_scene_fixed_camera_window")
        else:
            result.update(status="reject", reason="static_or_nearly_static")
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
        windows = [
            analyse_window(
                path,
                start,
                screening,
                allow_static_scene=bool(source.get("allow_static_scene", False)),
            )
            for start in starts
        ]
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
            "rule": "dynamic_or_explicitly_allowed_static_scenes_from_stationary_360_cameras",
            "privacy_rule": "360+x uses official face blurring and is REF-only",
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


def action_curve(time_seconds: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    """根据逐动作时间窗生成连续曲线，不再把 15 秒机械均分。

    所有参数都随 case 写入 ``CASE_PLAN.json``：

    ``start_seconds`` / ``end_seconds``
        动作真实生效时间窗。不同动作的起止点互相错开并大量重叠。
    ``profile``
        ``linear`` 持续匀速；``smooth`` 平滑接力；``whip`` 在窗口中段集中完成，
        形成甩镜；``swing`` 在总体前进上叠加变速/反向；``overshoot`` 越过目标后
        回摆；``pulse`` 运动后返回起点。
    ``speed_multiplier``
        不直接裁掉尾部，而是调节 whip 陡峭度并在计划阶段影响动作时长。这样调快
        不会产生“动作先做完、剩余时间静止”的平台。
    """

    start = float(item["start_seconds"])
    end = float(item["end_seconds"])
    if end <= start:
        raise ValueError(f"动作时间窗无效：{item}")
    local = (time_seconds - start) / (end - start)
    inside = (local >= 0.0) & (local <= 1.0)
    clipped = np.clip(local, 0.0, 1.0)
    profile = str(item["profile"])
    speed = float(item["speed_multiplier"])
    strength = float(item.get("curve_strength", 0.0))
    cycles = float(item.get("oscillation_cycles", 1.0))

    if profile == "linear":
        progress = clipped
    elif profile == "smooth":
        progress = smoothstep(clipped)
    elif profile == "whip":
        # tanh 归一化后严格从 0 到 1；倍率越大，中段角速度越高。
        sharpness = 3.0 + 2.2 * speed
        denominator = 2.0 * math.tanh(sharpness * 0.5)
        progress = (
            np.tanh(sharpness * (clipped - 0.5)) + math.tanh(sharpness * 0.5)
        ) / denominator
    elif profile == "swing":
        # 端点仍为 0/1，但中间可以多次加速、减速甚至短暂反向。
        progress = clipped + strength * np.sin(2.0 * np.pi * cycles * clipped) / (
            2.0 * np.pi * max(cycles, 1e-6)
        )
    elif profile == "overshoot":
        progress = clipped + strength * np.sin(np.pi * clipped)
    elif profile == "pulse":
        progress = np.zeros_like(clipped)
        progress[inside] = np.sin(np.pi * clipped[inside])
        # pulse 在窗口结束后返回 0，而不是保持 1。
        progress[local > 1.0] = 0.0
    else:
        raise ValueError(f"未知速度曲线：{profile}")
    return progress.astype(np.float32)


def longest_true_run(values: np.ndarray) -> int:
    """返回布尔数组中最长连续 True 帧数，用于检测肉眼可见停顿。"""

    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


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
    time_seconds = normalized_time * float(project["duration_seconds"])
    base_hfov = base_hfov_for_resolution(width, height, settings)

    raw_yaw = np.zeros(frame_count, np.float32)
    raw_pitch = np.zeros(frame_count, np.float32)
    raw_roll = np.zeros(frame_count, np.float32)
    raw_zoom = np.zeros(frame_count, np.float32)

    actions = case.get("actions", [])
    for item in actions:
        action = item["type"]
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"动作不在 V2 白名单：{action}")
        curve = action_curve(time_seconds, item)
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
                    # Roll 绕光轴旋转，360° 后回到同一姿态；它不会像 yaw 一样把
                    # 视角中心推向 ERP 左右边界，因此接缝缩放只作用于 yaw/pitch。
                    float(roll[frame_index]),
                )
                max_abs_longitude = max(max_abs_longitude, float(np.max(np.abs(longitudes))))
            if max_abs_longitude <= longitude_limit:
                break
            seam_scale *= 0.9
        else:
            raise RuntimeError(f"轨迹无法避开 ERP 接缝：{case['id']}")
        yaw *= seam_scale
        pitch *= seam_scale

    # 用逐帧角度/FOV 差计算真实合成速度，而不是只看单动作的名义参数。只要不同
    # 轴互相抵消、速度曲线尾部变平或动作时间窗有空洞，这里都会直接暴露。
    speed_components = np.stack(
        [np.diff(yaw), np.diff(pitch), np.diff(roll), 1.5 * np.diff(hfov)],
        axis=1,
    )
    combined_speed = np.linalg.norm(speed_components, axis=1) * fps
    stationary_threshold = float(settings["stationary_speed_threshold_deg_per_second"])
    stationary = combined_speed < stationary_threshold
    max_stationary_run = longest_true_run(stationary)
    speed_report = {
        "threshold_deg_per_second": round(stationary_threshold, 6),
        "min_deg_per_second": round(float(combined_speed.min()), 6),
        "median_deg_per_second": round(float(np.median(combined_speed)), 6),
        "p90_deg_per_second": round(float(np.percentile(combined_speed, 90)), 6),
        "p99_deg_per_second": round(float(np.percentile(combined_speed, 99)), 6),
        "max_deg_per_second": round(float(combined_speed.max(initial=0.0)), 6),
        "stationary_frame_fraction": round(float(np.mean(stationary)), 6),
        "max_stationary_run_frames": int(max_stationary_run),
        "continuous_motion": max_stationary_run <= int(settings["max_stationary_run_frames"]),
    }
    active_counts = np.zeros(frame_count, np.int16)
    for item in actions:
        active_counts += (
            (time_seconds >= float(item["start_seconds"]))
            & (time_seconds <= float(item["end_seconds"]))
        ).astype(np.int16)
    concurrency_report = {
        "max_concurrent_actions": int(active_counts.max(initial=0)),
        "fraction_at_least_2": round(float(np.mean(active_counts >= 2)), 6),
        "fraction_at_least_3": round(float(np.mean(active_counts >= 3)), 6),
        "frames_at_least_3": int(np.sum(active_counts >= 3)),
    }

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
                "seam_yaw_pitch": round(seam_scale, 6),
            },
            "max_abs_sampled_longitude_deg": round(max_abs_longitude, 6),
            "seam_longitude_limit_deg": round(longitude_limit, 6),
            "seam_safe": case["kind"] != "combo" or max_abs_longitude <= longitude_limit,
            "speed": speed_report,
            "concurrency": concurrency_report,
            "timeline": [
                {
                    "type": item["type"],
                    "start_seconds": item["start_seconds"],
                    "end_seconds": item["end_seconds"],
                    "duration_seconds": item["duration_seconds"],
                    "profile": item["profile"],
                    "nominal_speed_deg_per_second": item["nominal_speed_deg_per_second"],
                }
                for item in actions
            ],
        },
    }


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
        "allowed_sides": list(source.get("allowed_sides", ["ref", "target"])),
        "privacy_face_blurred": bool(source.get("privacy_face_blurred", False)),
        "synthetic_or_3d": bool(source.get("synthetic_or_3d", False)),
        "scene_policy": window.get("reason", "dynamic_fixed_camera_window"),
    }


class BalancedWindowAllocator:
    """优先使用尚未用过的 15 秒窗口，并避免连续重复同一个源文件。

    随机抽样很容易让少数文件被连续抽中。这里先把“源文件 × 可用窗口”展开成
    唯一条目，每轮只从使用次数最低的条目里随机选；所有窗口用过一遍后才进入
    下一轮。这样仍然随机，但不会产生无意义的重复偏置。
    """

    def __init__(self, sources: list[dict[str, Any]], side: str, rng: random.Random):
        self.side = side
        self.rng = rng
        self.entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for source in sources:
            if side not in source.get("allowed_sides", ["ref", "target"]):
                continue
            for window in source["usable_windows"]:
                self.entries.append((source, window))
        if not self.entries:
            raise RuntimeError(f"{side} 没有可分配的合格素材窗口")
        self.usage = [0 for _ in self.entries]
        self.last_file: str | None = None

    def take(self) -> dict[str, Any]:
        minimum = min(self.usage)
        candidates = [index for index, count in enumerate(self.usage) if count == minimum]
        non_repeating = [
            index for index in candidates if self.entries[index][0]["file"] != self.last_file
        ]
        if non_repeating:
            candidates = non_repeating
        selected = self.rng.choice(candidates)
        self.usage[selected] += 1
        source, window = self.entries[selected]
        self.last_file = source["file"]
        return public_source_record(source, window)

    def report(self) -> dict[str, Any]:
        by_file: dict[str, int] = {}
        by_window: dict[str, int] = {}
        for (source, window), count in zip(self.entries, self.usage):
            key = f"{source['dataset']}/{source['file']}@{float(window['start_seconds']):.3f}"
            by_window[key] = count
            file_key = f"{source['dataset']}/{source['file']}"
            by_file[file_key] = by_file.get(file_key, 0) + count
        return {
            "side": self.side,
            "source_file_count": len(by_file),
            "source_window_count": len(by_window),
            "maximum_exact_window_reuse": max(by_window.values(), default=0),
            "by_file": by_file,
            "by_window": by_window,
        }


def apply_case_source_overrides(
    cases: list[dict[str, Any]],
    overrides: dict[str, Any],
) -> None:
    """只替换人工复核确认有额外源运动的单侧窗口。

    ``from_case`` 必须指向当前确定性计划中已经人工确认正常的 case。复制的是完整
    公开源记录，不改变被替换 case 的动作、轨迹、分辨率或另一侧素材。
    """

    by_id = {case["id"]: case for case in cases}
    for case_id, side_rules in overrides.items():
        if case_id not in by_id:
            raise ValueError(f"素材覆盖目标不存在：{case_id}")
        target_case = by_id[case_id]
        for side, rule in side_rules.items():
            if side not in {"ref", "target"}:
                raise ValueError(f"素材覆盖 side 非法：{case_id}/{side}")
            donor_id = str(rule["from_case"])
            if donor_id not in by_id:
                raise ValueError(f"素材覆盖 donor 不存在：{donor_id}")
            target_case[side] = dict(by_id[donor_id][side])
            target_case.setdefault("source_overrides", {})[side] = {
                "from_case": donor_id,
                "reason": str(rule.get("reason", "人工复核替换")),
            }


def allocation_report_from_cases(cases: list[dict[str, Any]], side: str) -> dict[str, Any]:
    """从应用人工覆盖后的最终 case 重新统计真实素材复用情况。"""

    by_file: dict[str, int] = {}
    by_window: dict[str, int] = {}
    for case in cases:
        source = case[side]
        file_key = f"{source['dataset']}/{source['file']}"
        window_key = f"{file_key}@{float(source['start_seconds']):.3f}"
        by_file[file_key] = by_file.get(file_key, 0) + 1
        by_window[window_key] = by_window.get(window_key, 0) + 1
    return {
        "side": side,
        "source_file_count": len(by_file),
        "source_window_count": len(by_window),
        "maximum_exact_window_reuse": max(by_window.values(), default=0),
        "by_file": by_file,
        "by_window": by_window,
    }


def action_axis(action: str) -> str:
    if action.startswith("pan_"):
        return "yaw"
    if action.startswith("tilt_"):
        return "pitch"
    if action.startswith("roll_"):
        return "roll"
    if action.startswith("zoom_"):
        return "zoom"
    raise ValueError(f"未知动作轴：{action}")


def choose_action_types(rng: random.Random, action_count: int, recipe_index: int) -> list[str]:
    """只让 4/45 个 recipe 含 roll；加上 10 个特殊投影，总 roll case 为 18%。"""

    required: list[str] = []
    roll_recipe = recipe_index in {0, 12, 24, 36}
    if roll_recipe:
        required.append(rng.choice(["roll_cw", "roll_ccw"]))
    if recipe_index % 3 == 0:
        required.append(rng.choice(["pan_left", "pan_right", "tilt_up", "tilt_down"]))
    if recipe_index % 4 == 1:
        required.append(rng.choice(["zoom_in", "zoom_out"]))
    required = list(dict.fromkeys(required))[:action_count]
    # 非 roll recipe 完全排除顺/逆时针滚转；roll recipe 也只保留已选中的一个方向。
    remaining = [
        action
        for action in ALLOWED_ACTIONS
        if action not in required and not action.startswith("roll_")
    ]
    rng.shuffle(remaining)
    return required + remaining[: action_count - len(required)]


def sample_speed_multiplier(rng: random.Random, settings: dict[str, Any]) -> tuple[float, str]:
    if rng.random() < float(settings["fast_action_probability"]):
        low, high = map(float, settings["fast_speed_multiplier_range"])
        return round(rng.uniform(low, high), 3), "fast"
    low, high = map(float, settings["speed_multiplier_range"])
    return round(rng.uniform(low, high), 3), "varied"


def sample_duration_seconds(
    rng: random.Random,
    settings: dict[str, Any],
    speed_multiplier: float,
) -> tuple[float, str]:
    draw = rng.random()
    fast_probability = float(settings["fast_action_probability"])
    medium_probability = float(settings["medium_action_probability"])
    if draw < fast_probability:
        low, high = map(float, settings["short_duration_seconds"])
        bucket = "short_fast"
    elif draw < fast_probability + medium_probability:
        low, high = map(float, settings["medium_duration_seconds"])
        bucket = "medium"
    else:
        low, high = map(float, settings["long_duration_seconds"])
        bucket = "long"
    base = rng.uniform(low, high)
    # 大倍率会缩短窗口，但保留完整曲线，不会像旧实现那样提前做完后静止。
    duration = base / max(0.85, speed_multiplier ** 0.35)
    return round(min(13.5, max(0.5, duration)), 3), bucket


def enforce_simultaneous_overlap(
    actions: list[dict[str, Any]],
    rng: random.Random,
    settings: dict[str, Any],
) -> bool:
    """让多个非 backbone 动作共享一段明显的同时运动区间。

    动作数 >=3 时至少选择两条非 backbone；动作更多时按配置比例增加。它们与
    全程 backbone 同时存在，因此共享区间内至少 3 种运镜同时叠加。
    """

    non_backbone = [item for item in actions if not item.get("is_backbone")]
    if len(non_backbone) < 2:
        return False
    selected_count = max(
        2,
        min(
            len(non_backbone),
            math.ceil(len(non_backbone) * float(settings["simultaneous_action_fraction"])),
        ),
    )
    selected = rng.sample(non_backbone, selected_count)
    low, high = map(float, settings["simultaneous_overlap_seconds"])
    overlap = rng.uniform(low, high)
    half = overlap / 2.0
    center = rng.uniform(0.5 + half, 14.5 - half)
    cluster_start = center - half
    cluster_end = center + half

    for item in selected:
        item["start_seconds"] = round(min(float(item["start_seconds"]), cluster_start), 3)
        item["end_seconds"] = round(max(float(item["end_seconds"]), cluster_end), 3)
        item["duration_seconds"] = round(
            float(item["end_seconds"]) - float(item["start_seconds"]), 3
        )
        strength = float(item.get("curve_strength", 0.0))
        path_factor = 2.0 if item["profile"] == "pulse" else 1.0 + 0.45 * strength
        item["nominal_speed_deg_per_second"] = round(
            abs(float(item["amplitude_deg"])) * path_factor
            / max(float(item["duration_seconds"]), 1e-6),
            3,
        )
        item["window_bucket"] = f"{item['window_bucket']}+simultaneous_cluster"
    return True


def build_case_actions(
    rng: random.Random,
    action_types: list[str],
    recipe_index: int,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """生成一个连续、错峰叠加、速度非统一的 15 秒动作表。"""

    duration_total = 15.0
    spin_candidates = [i for i, action in enumerate(action_types) if action.startswith("roll_")]
    whip_candidates = [
        i for i, action in enumerate(action_types) if action.startswith("pan_") or action.startswith("tilt_")
    ]
    spin_index: int | None = None
    if spin_candidates:
        spin_index = rng.choice(spin_candidates)
    whip_index: int | None = None
    if whip_candidates and (
        recipe_index % 3 == 0 or rng.random() < float(settings["whip_probability"])
    ):
        whip_index = rng.choice(whip_candidates)
    large_whip = whip_index is not None and (
        recipe_index % 9 == 0 or rng.random() < float(settings["large_whip_probability"])
    )

    non_zoom = [i for i, action in enumerate(action_types) if not action.startswith("zoom_")]
    backbone_index = spin_index if spin_index is not None else rng.choice(non_zoom or list(range(len(action_types))))
    starts: list[float] = [0.0]
    actions: list[dict[str, Any]] = []
    tags = ["continuous_layered_timeline", "fast_large_range_bias"]
    if spin_index is not None:
        tags.append("spin_360")
    if whip_index is not None:
        tags.append("whip_pan" if action_types[whip_index].startswith("pan_") else "whip_tilt")
    if large_whip:
        tags.append("large_range_whip")

    for index, action in enumerate(action_types):
        speed_multiplier, speed_bucket = sample_speed_multiplier(rng, settings)
        axis = action_axis(action)
        is_backbone = index == backbone_index
        is_whip = index == whip_index and not is_backbone
        is_spin = index == spin_index

        if is_backbone:
            start = float(settings["backbone_start_seconds"])
            end = float(settings["backbone_end_seconds"])
            window_bucket = "full_length_backbone"
            profile = "linear" if is_spin or rng.random() < 0.62 else "swing"
        elif is_whip:
            low, high = map(float, settings["whip_duration_seconds"])
            action_duration = rng.uniform(low, high)
            if large_whip:
                action_duration *= rng.uniform(0.72, 0.95)
            start = rng.uniform(0.35, max(0.36, duration_total - action_duration - 0.05))
            end = min(duration_total, start + action_duration)
            window_bucket = "large_whip" if large_whip else "whip"
            profile = "whip"
            speed_multiplier = max(speed_multiplier, rng.uniform(2.4, 3.6))
            speed_bucket = "fast"
        else:
            action_duration, window_bucket = sample_duration_seconds(rng, settings, speed_multiplier)
            latest = min(
                float(settings["action_start_latest_seconds"]),
                duration_total - action_duration,
            )
            start = rng.uniform(0.35, max(0.36, latest))
            # 尽量避免多个动作在完全相同的时刻开始。
            for _ in range(12):
                if all(
                    abs(start - previous) >= float(settings["min_start_separation_seconds"])
                    for previous in starts
                ):
                    break
                start = rng.uniform(0.35, max(0.36, latest))
            end = min(duration_total, start + action_duration)
            if axis == "zoom":
                profile = rng.choice(["smooth", "pulse", "swing"])
            else:
                profile = rng.choices(
                    ["linear", "smooth", "swing", "overshoot", "pulse", "whip"],
                    weights=[17, 16, 25, 15, 10, 17],
                    k=1,
                )[0]
        starts.append(float(start))

        if is_spin:
            low, high = map(float, settings["spin_360_amplitude_deg"])
            amplitude = round(rng.uniform(low, high), 3)
        elif is_whip and axis == "yaw":
            low, high = map(float, settings["whip_pan_amplitude_deg"])
            amplitude = round(rng.uniform(low, high), 3)
        elif is_whip and axis == "pitch":
            low, high = map(float, settings["whip_tilt_amplitude_deg"])
            amplitude = round(rng.uniform(low, high), 3)
        else:
            amplitude = amplitude_for_action(action, rng, settings)

        action_duration = max(float(end) - float(start), 1e-6)
        curve_strength = 0.0
        oscillation_cycles = 1.0
        if profile == "swing":
            curve_strength = round(rng.uniform(1.15, 2.40), 3)
            oscillation_cycles = round(rng.uniform(1.1, 3.4), 3)
        elif profile == "overshoot":
            curve_strength = round(rng.uniform(0.38, 0.92), 3)
        elif profile == "pulse":
            oscillation_cycles = 0.5
        path_factor = 2.0 if profile == "pulse" else 1.0 + 0.45 * curve_strength
        nominal_speed = abs(amplitude) * path_factor / action_duration
        actions.append(
            {
                "type": action,
                "axis": axis,
                "amplitude_deg": round(float(amplitude), 3),
                "start_seconds": round(float(start), 3),
                "end_seconds": round(float(end), 3),
                "duration_seconds": round(action_duration, 3),
                "speed_multiplier": round(float(speed_multiplier), 3),
                "speed_bucket": speed_bucket,
                "window_bucket": window_bucket,
                "profile": profile,
                "curve_strength": curve_strength,
                "oscillation_cycles": oscillation_cycles,
                "nominal_speed_deg_per_second": round(float(nominal_speed), 3),
                "is_backbone": is_backbone,
                "parameter_hint": (
                    "start/end 控制时机；amplitude 控制幅度；speed_multiplier 与 duration 控制快慢；"
                    "profile/curve_strength/oscillation_cycles 控制变速、回摆和不规则程度"
                ),
            }
        )
    if enforce_simultaneous_overlap(actions, rng, settings):
        tags.append("simultaneous_overlap_cluster")
    actions.sort(key=lambda item: (float(item["start_seconds"]), not bool(item["is_backbone"])))
    return actions, tags


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
    ref_allocator = BalancedWindowAllocator(pool, "ref", rng)
    target_allocator = BalancedWindowAllocator(pool, "target", rng)

    # 45 个配方平均覆盖动作数 2、3、4、5、6：每种动作数恰好 9 个配方。每个配方
    # 生成两个独立混合时序变体，动作类型相同，但时间窗/幅度/速度/素材都重新采样。
    for recipe_index in range(int(config["project"]["combo_recipe_count"])):
        action_count = 2 + (recipe_index % 5)
        action_types = choose_action_types(rng, action_count, recipe_index)
        recipe = {
            "id": f"recipe_{recipe_index + 1:02d}",
            "display_name": None,
            "origin": "randomized_continuous_layered_timeline",
            "action_count": action_count,
            "action_types": action_types,
        }
        recipes.append(recipe)
        for variant_index, mode in enumerate(("hybrid_a", "hybrid_b")):
            # 极少数随机同轴组合可能发生速度抵消；最多重采样 40 次，直到真实逐帧
            # 速度验证确认无停顿且接缝安全。
            for _ in range(40):
                actions, tags = build_case_actions(
                    rng, action_types, recipe_index, config["trajectory"]
                )
                case = {
                    "id": f"{recipe['id']}_{'a' if mode == 'hybrid_a' else 'b'}",
                    "kind": "combo",
                    "mode": mode,
                    "recipe_id": recipe["id"],
                    "display_name": recipe["display_name"],
                    "origin": recipe["origin"],
                    "action_count": action_count,
                    "actions": actions,
                    "showcase_tags": tags,
                    "resolution": list(
                        resolutions[(recipe_index * 2 + variant_index) % len(resolutions)]
                    ),
                    "duration_seconds": 15.0,
                    "fps": int(config["project"]["fps"]),
                }
                try:
                    trajectory = build_trajectory(case, config)
                except RuntimeError:
                    continue
                if trajectory["report"]["speed"]["continuous_motion"]:
                    case["ref"] = ref_allocator.take()
                    case["target"] = target_allocator.take()
                    case["trajectory_report"] = trajectory["report"]
                    cases.append(case)
                    break
            else:
                raise RuntimeError(f"{recipe['id']} {mode} 无法生成无停顿安全轨迹")

    # 特殊投影现在明确叠加持续顺/逆时针滚转和轻微呼吸 Zoom，但仍不混入普通
    # rectilinear 组合类别。
    for kind, count in (
        ("tiny_planet", int(config["project"]["tiny_planet_cases"])),
        ("rabbit_hole", int(config["project"]["rabbit_hole_cases"])),
    ):
        for index in range(count):
            direction = rng.choice(["roll_cw", "roll_ccw"])
            roll_low, roll_high = map(float, config["trajectory"]["special_roll_total_deg"])
            zoom_low, zoom_high = map(float, config["trajectory"]["special_zoom_percent"])
            cycle_low, cycle_high = map(float, config["trajectory"]["special_zoom_cycles"])
            roll_total = round(rng.uniform(roll_low, roll_high), 3)
            zoom_percent = round(rng.uniform(zoom_low, zoom_high), 5)
            zoom_cycles = round(rng.uniform(cycle_low, cycle_high), 3)
            case = {
                "id": f"{kind}_{index + 1:02d}",
                "kind": kind,
                "mode": "special_motion",
                "recipe_id": None,
                "action_count": 2,
                "actions": [
                    {
                        "type": direction,
                        "amplitude_deg": roll_total,
                        "start_seconds": 0.0,
                        "end_seconds": 15.0,
                        "duration_seconds": 15.0,
                        "profile": "linear",
                        "speed_multiplier": round(roll_total / 360.0, 3),
                    },
                    {
                        "type": rng.choice(["zoom_in", "zoom_out"]),
                        "amplitude_percent": zoom_percent,
                        "start_seconds": 0.0,
                        "end_seconds": 15.0,
                        "duration_seconds": 15.0,
                        "profile": "sine_breathing",
                        "speed_multiplier": zoom_cycles,
                    },
                ],
                "showcase_tags": ["special_projection", "continuous_roll", "light_zoom"],
                "resolution": list(resolutions[(90 + index + (0 if kind == 'tiny_planet' else 5)) % len(resolutions)]),
                "duration_seconds": 15.0,
                "fps": int(config["project"]["fps"]),
                "ref": ref_allocator.take(),
                "target": target_allocator.take(),
                "projection_parameters": {
                    "stereographic_scale": round(0.92 + 0.04 * index, 3),
                    "seam_orientation_deg": float(-144 + 72 * index),
                    "roll_direction": direction,
                    "roll_total_deg": roll_total,
                    "zoom_percent": zoom_percent,
                    "zoom_cycles": zoom_cycles,
                    "zoom_phase_radians": round(rng.uniform(0.0, 2.0 * math.pi), 6),
                    "note": "特殊投影使用完整 360°，并叠加持续滚转与轻微呼吸 Zoom",
                },
                "trajectory_report": {
                    "frame_count": int(15 * config["project"]["fps"]),
                    "seam_safe": None,
                    "roll_total_deg": roll_total,
                    "roll_speed_deg_per_second": round(roll_total / 15.0, 6),
                    "zoom_scale_range": [round(1.0 - zoom_percent, 6), round(1.0 + zoom_percent, 6)],
                    "speed": {
                        "continuous_motion": True,
                        "max_stationary_run_frames": 0,
                        "median_deg_per_second": round(roll_total / 15.0, 6),
                    },
                    "note": "全 360° 特殊投影无法像普通透视视角一样完全排除 ERP 接缝",
                },
            }
            cases.append(case)

    apply_case_source_overrides(cases, config.get("case_source_overrides", {}))

    # 网页按 recipe 相邻展示两个独立混合时序变体；特殊投影放在最后。
    payload = {
        "schema_version": 3,
        "random_seed": int(config["project"]["random_seed"]),
        "summary": {
            "total_cases": len(cases),
            "combo_cases": sum(case["kind"] == "combo" for case in cases),
            "hybrid_a_cases": sum(case["mode"] == "hybrid_a" for case in cases),
            "hybrid_b_cases": sum(case["mode"] == "hybrid_b" for case in cases),
            "tiny_planet_cases": sum(case["kind"] == "tiny_planet" for case in cases),
            "rabbit_hole_cases": sum(case["kind"] == "rabbit_hole" for case in cases),
            "videos": len(cases) * 2,
            "duration_seconds_each": 15.0,
            "max_pixels": int(config["project"]["max_pixels"]),
            "pure_sequential_cases": 0,
            "pure_simultaneous_cases": 0,
            "continuous_motion_required": True,
        },
        "source_allocation": {
            "ref": allocation_report_from_cases(cases, "ref"),
            "target": allocation_report_from_cases(cases, "target"),
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
                # 小行星/兔子洞不再使用一张静态采样表：orientation 逐帧滚转，scale
                # 按正弦轻微呼吸，实现持续顺/逆时针旋转 + 轻微 Zoom。
                progress = frame_index / max(frame_count - 1, 1)
                roll_sign = 1.0 if params["roll_direction"] == "roll_cw" else -1.0
                orientation = (
                    float(params["seam_orientation_deg"])
                    + roll_sign * float(params["roll_total_deg"]) * progress
                )
                zoom_wave = math.sin(
                    2.0 * math.pi * float(params["zoom_cycles"]) * progress
                    + float(params["zoom_phase_radians"])
                )
                scale = float(params["stereographic_scale"]) * (
                    1.0 + float(params["zoom_percent"]) * zoom_wave
                )
                ref_maps = stereographic_maps(
                    (width, height),
                    (ref_reader.width, ref_reader.height),
                    case["kind"],
                    scale,
                    orientation,
                )
                target_maps = stereographic_maps(
                    (width, height),
                    (target_reader.width, target_reader.height),
                    case["kind"],
                    scale,
                    orientation,
                )

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
        "hybrid_a": sum(case["mode"] == "hybrid_a" for case in cases),
        "hybrid_b": sum(case["mode"] == "hybrid_b" for case in cases),
        "tiny_planet": sum(case["kind"] == "tiny_planet" for case in cases),
        "rabbit_hole": sum(case["kind"] == "rabbit_hole" for case in cases),
    }
    expected_counts = {"combo": 90, "hybrid_a": 45, "hybrid_b": 45, "tiny_planet": 5, "rabbit_hole": 5}
    if counts != expected_counts:
        errors.append(f"类目计数不正确：{counts}")

    # 每个 recipe 必须拥有两个混合时序变体；动作类型相同，但时间窗、幅度和速度
    # 应独立随机。纯 sequential/simultaneous 已经被本版删除。
    recipe_modes: dict[str, dict[str, dict[str, Any]]] = {}
    combo_cases = [case for case in cases if case["kind"] == "combo"]
    whip_case_count = 0
    large_whip_case_count = 0
    spin_360_case_count = 0
    combo_roll_case_count = 0
    simultaneous_three_plus_case_count = 0
    high_speed_case_count = 0
    speed_medians: list[float] = []
    source_files: dict[str, set[str]] = {"ref": set(), "target": set()}
    window_usage: dict[str, dict[str, int]] = {"ref": {}, "target": {}}
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
            speed = case["trajectory_report"].get("speed", {})
            if not speed.get("continuous_motion"):
                errors.append(f"{case['id']} 存在可见停顿：{speed}")
            if int(speed.get("max_stationary_run_frames", 999)) > int(
                config["trajectory"]["max_stationary_run_frames"]
            ):
                errors.append(f"{case['id']} 最长停顿帧数超限")
            speed_medians.append(float(speed.get("median_deg_per_second", 0.0)))
            if float(speed.get("p99_deg_per_second", 0.0)) >= float(
                config["trajectory"]["high_speed_threshold_deg_per_second"]
            ):
                high_speed_case_count += 1
            tags = set(case.get("showcase_tags", []))
            whip_case_count += int("whip_pan" in tags or "whip_tilt" in tags)
            large_whip_case_count += int("large_range_whip" in tags)
            spin_360_case_count += int("spin_360" in tags)
            combo_roll_case_count += int(
                any(item["type"].startswith("roll_") for item in case["actions"])
            )
            concurrency = case["trajectory_report"].get("concurrency", {})
            if int(case["action_count"]) >= 3:
                if int(concurrency.get("max_concurrent_actions", 0)) < 3:
                    errors.append(f"{case['id']} 没有至少 3 种运镜同时叠加")
                elif float(concurrency.get("fraction_at_least_3", 0.0)) < 0.15:
                    errors.append(f"{case['id']} 三动作同时叠加时长不足")
                else:
                    simultaneous_three_plus_case_count += 1

            actions = case["actions"]
            backbones = [item for item in actions if item.get("is_backbone")]
            if len(backbones) != 1:
                errors.append(f"{case['id']} 必须恰好有一个持续 backbone")
            elif (
                float(backbones[0]["start_seconds"]) != 0.0
                or float(backbones[0]["end_seconds"]) != 15.0
            ):
                errors.append(f"{case['id']} backbone 没有覆盖完整 15 秒")
            if not any(float(item["start_seconds"]) > 0.0 for item in actions):
                errors.append(f"{case['id']} 没有错峰动作")
            durations = [float(item["duration_seconds"]) for item in actions]
            if max(durations) - min(durations) < 0.25:
                errors.append(f"{case['id']} 动作时长缺少变化")
        else:
            action_types = {item["type"] for item in case["actions"]}
            if case["action_count"] != 2 or not any(
                item.startswith("roll_") for item in action_types
            ) or not any(item.startswith("zoom_") for item in action_types):
                errors.append(f"{case['id']} 特殊投影必须包含滚转 + 轻微 Zoom")
            params = case.get("projection_parameters", {})
            if not 200.0 <= float(params.get("roll_total_deg", 0.0)) <= 720.0:
                errors.append(f"{case['id']} 特殊投影滚转幅度不正确")
            if not 0.0 < float(params.get("zoom_percent", 0.0)) <= 0.08:
                errors.append(f"{case['id']} 特殊投影 Zoom 不应过大")

        for side in ("ref", "target"):
            source = case[side]
            source_files[side].add(f"{source['dataset']}/{source['file']}")
            window_key = (
                f"{source['dataset']}/{source['file']}@{float(source['start_seconds']):.3f}"
            )
            window_usage[side][window_key] = window_usage[side].get(window_key, 0) + 1
            if source.get("privacy_face_blurred") and side != "ref":
                errors.append(f"{case['id']} 人脸模糊素材只能放 REF")
            if source["dataset"] == "360x" and side != "ref":
                errors.append(f"{case['id']} 360+x 被错误放入 TARGET")
            if source.get("synthetic_or_3d") and side != "ref":
                errors.append(f"{case['id']} 3D/动画素材只能放 REF")

    for recipe_id, modes in recipe_modes.items():
        if set(modes) != {"hybrid_a", "hybrid_b"}:
            errors.append(f"{recipe_id} 缺少 hybrid_a 或 hybrid_b")
        else:
            types_a = sorted(item["type"] for item in modes["hybrid_a"]["actions"])
            types_b = sorted(item["type"] for item in modes["hybrid_b"]["actions"])
            if types_a != types_b:
                errors.append(f"{recipe_id} 两个变体的动作类型不一致")
            if modes["hybrid_a"]["actions"] == modes["hybrid_b"]["actions"]:
                errors.append(f"{recipe_id} 两个变体不应复用完全相同参数")

    if any(case["mode"] in {"sequential", "simultaneous"} for case in cases):
        errors.append("仍存在纯顺序或纯叠加 case")
    if whip_case_count < 24:
        errors.append(f"甩镜覆盖不足：{whip_case_count}")
    if large_whip_case_count < 8:
        errors.append(f"大范围甩镜覆盖不足：{large_whip_case_count}")
    total_roll_case_count = sum(
        int(any(item["type"].startswith("roll_") for item in case["actions"]))
        for case in cases
    )
    if spin_360_case_count < 6:
        errors.append(f"360°滚转环绕覆盖不足：{spin_360_case_count}")
    if not 15 <= total_roll_case_count <= 20:
        errors.append(f"含 roll 的 case 应为 15%～20%，当前 {total_roll_case_count}/100")
    if simultaneous_three_plus_case_count < 72:
        errors.append(
            f"三动作同时叠加覆盖不足：{simultaneous_three_plus_case_count}/72"
        )
    if high_speed_case_count < 63:
        errors.append(f"快速运镜占比不足：{high_speed_case_count}/90")
    if len({round(value, 1) for value in speed_medians}) < 35:
        errors.append("组合运镜速度分布过于统一")
    if len(source_files["ref"]) < 7:
        errors.append(f"REF 源文件种类过少：{len(source_files['ref'])}")
    if len(source_files["target"]) < 5:
        errors.append(f"TARGET 源文件种类过少：{len(source_files['target'])}")
    for side in ("ref", "target"):
        maximum_reuse = max(window_usage[side].values(), default=0)
        if maximum_reuse > 5:
            errors.append(f"{side} 单一 15 秒窗口重复过多：{maximum_reuse}")

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
        "schema_version": 3,
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
            "pure_sequential_cases": 0,
            "continuous_motion_every_combo": True,
            "fast_motion_bias": {
                "high_speed_threshold_deg_per_second": float(
                    config["trajectory"]["high_speed_threshold_deg_per_second"]
                ),
                "high_speed_cases": high_speed_case_count,
                "combo_cases": len(combo_cases),
                "whip_cases": whip_case_count,
                "large_whip_cases": large_whip_case_count,
                "spin_360_cases": spin_360_case_count,
                "combo_roll_cases": combo_roll_case_count,
                "total_roll_cases": total_roll_case_count,
            },
            "simultaneous_overlap": {
                "three_plus_concurrent_cases": simultaneous_three_plus_case_count,
                "eligible_cases": sum(
                    int(case["action_count"]) >= 3 for case in combo_cases
                ),
            },
            "source_role_policy": (
                "360+x face-blurred and synthetic/3D material are REF-only; "
                "TARGET contains real scenes only"
            ),
            "source_diversity": {
                "ref_source_files": len(source_files["ref"]),
                "target_source_files": len(source_files["target"]),
                "ref_unique_windows": len(window_usage["ref"]),
                "target_unique_windows": len(window_usage["target"]),
                "ref_maximum_window_reuse": max(window_usage["ref"].values(), default=0),
                "target_maximum_window_reuse": max(window_usage["target"].values(), default=0),
            },
            "special_projection_motion": "continuous roll plus light breathing zoom",
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

    audit = subparsers.add_parser("audit", help="筛选固定机位动态/允许静态的全景素材")
    audit.add_argument("--source-root-360x", help="覆盖 360+x 素材目录")
    audit.add_argument("--source-root-commons", help="覆盖 Wikimedia 素材目录")
    audit.add_argument("--source-root-commons-still", help="覆盖 Wikimedia 真实静态全景目录")

    subparsers.add_parser("plan", help="生成 100-case 确定性计划")

    render = subparsers.add_parser("render", help="渲染 ref/target MP4")
    render.add_argument("--workers", type=int, default=1, help="并行 case 数；推荐 1～2")
    render.add_argument("--limit", type=int, help="只渲染计划开头 N 条，用于冒烟测试")
    render.add_argument("--case", action="append", help="只渲染指定 case id，可重复传入")
    render.add_argument("--overwrite", action="store_true", help="覆盖已经完整存在的 case")
    render.add_argument("--source-root-360x", help="本次渲染覆盖 360+x 素材目录")
    render.add_argument("--source-root-commons", help="本次渲染覆盖 Wikimedia 素材目录")
    render.add_argument("--source-root-commons-still", help="本次渲染覆盖真实静态全景目录")

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
