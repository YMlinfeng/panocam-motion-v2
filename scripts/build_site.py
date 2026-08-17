#!/usr/bin/env python3
"""从 CASE_PLAN / SOURCE_AUDIT 生成无需后端的静态展示网页。

本项目一共只有两个 Python 文件：生产主程序 ``pipeline.py`` 和本网页生成器。
网页不做二次转码；它直接播放生产阶段验证过的 H.264 MP4。每个 case 无论屏幕
宽度如何都保留 ref / target 左右两栏，不会为了省空间隐藏其中一路。
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = PROJECT_ROOT / "site"
PLAN_PATH = PROJECT_ROOT / "output" / "CASE_PLAN.json"
AUDIT_PATH = PROJECT_ROOT / "output" / "SOURCE_AUDIT.json"
VALIDATION_PATH = PROJECT_ROOT / "output" / "VALIDATION.json"


ACTION_ZH = {
    "pan_left": "左摇",
    "pan_right": "右摇",
    "tilt_up": "上摇",
    "tilt_down": "下摇",
    "roll_cw": "顺时针滚转",
    "roll_ccw": "逆时针滚转",
    "zoom_in": "小幅 Zoom In",
    "zoom_out": "小幅 Zoom Out",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def video_panel(case: dict[str, Any], side: str) -> str:
    source = case[side]
    label = "REF · 参考场景" if side == "ref" else "TARGET · 目标场景"
    src = f"media/cases/{escape(case['id'])}/{side}.mp4"
    poster = f"media/cases/{escape(case['id'])}/{side}.jpg"
    width, height = case["resolution"]
    return f"""
      <figure class="video-panel">
        <figcaption><b>{label}</b><span>{escape(source['dataset'])}</span></figcaption>
        <video controls playsinline preload="metadata" src="{src}" poster="{poster}" style="aspect-ratio:{width}/{height}"></video>
        <div class="source-line" title="{escape(source['file'])}">
          {escape(source['file'])} · t={float(source['start_seconds']):.2f}s
        </div>
      </figure>"""


def case_card(case: dict[str, Any]) -> str:
    width, height = case["resolution"]
    if case["kind"] == "combo":
        mode_zh = "混合时序 A" if case["mode"] == "hybrid_a" else "混合时序 B"
        actions = " ｜ ".join(
            f"{ACTION_ZH[item['type']]} {float(item['start_seconds']):.1f}–{float(item['end_seconds']):.1f}s"
            f" · {item['profile']} · {float(item['nominal_speed_deg_per_second']):.1f}°/s"
            for item in case["actions"]
        )
        title = f"{mode_zh} · 连续错峰叠加 · {case['action_count']} 动作"
        kind_class = case["mode"]
        speed = case["trajectory_report"]["speed"]
        speed_badge = (
            f"<b>中位 {float(speed['median_deg_per_second']):.1f}°/s</b>"
            f"<b>P99 {float(speed['p99_deg_per_second']):.1f}°/s</b>"
        )
    else:
        title = "小行星 + 持续滚转/轻微 Zoom" if case["kind"] == "tiny_planet" else "兔子洞 + 持续滚转/轻微 Zoom"
        params = case["projection_parameters"]
        direction = "顺时针" if params["roll_direction"] == "roll_cw" else "逆时针"
        actions = (
            f"{direction}滚转 {float(params['roll_total_deg']):.1f}° / 15s ｜ "
            f"呼吸 Zoom ±{float(params['zoom_percent']) * 100:.1f}% ｜ "
            f"{float(params['zoom_cycles']):.2f} cycles"
        )
        kind_class = "special"
        speed_badge = f"<b>滚转 {float(case['trajectory_report']['roll_speed_deg_per_second']):.1f}°/s</b>"
    return f"""
    <article id="{escape(case['id'])}" class="case-card {kind_class}" data-mode="{escape(case['mode'])}" data-actions="{case['action_count']}">
      <header>
        <div><span class="case-id">{escape(case['id'])}</span><h3>{title}</h3></div>
        <div class="badges"><b>15.00s</b><b>{width}×{height}</b><b>{width * height:,} px</b>{speed_badge}</div>
      </header>
      <p class="actions">{escape(actions)}</p>
      <div class="video-pair" data-sync-pair>
        {video_panel(case, 'ref')}
        {video_panel(case, 'target')}
      </div>
    </article>"""


def build_html(plan: dict[str, Any], audit: dict[str, Any], validation: dict[str, Any] | None) -> str:
    recipes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    specials: list[dict[str, Any]] = []
    for case in plan["cases"]:
        if case["kind"] == "combo":
            recipes[case["recipe_id"]].append(case)
        else:
            specials.append(case)

    recipe_sections = []
    for recipe_id, cases in recipes.items():
        cases.sort(key=lambda item: 0 if item["mode"] == "hybrid_a" else 1)
        action_text = " + ".join(ACTION_ZH[item["type"]] for item in cases[0]["actions"])
        recipe_title = cases[0].get("display_name") or recipe_id
        recipe_sections.append(f"""
        <section class="recipe" data-recipe data-action-count="{cases[0]['action_count']}">
          <div class="recipe-head">
            <h2>{escape(recipe_title)}</h2>
            <p>{escape(action_text)}</p>
          </div>
          {''.join(case_card(case) for case in cases)}
        </section>""")

    special_section = f"""
      <section class="special-section" id="specials">
        <div class="section-title"><h2>特殊投影</h2></div>
        {''.join(case_card(case) for case in specials)}
      </section>"""

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Panocam Motion V2 · 100 Cross-Pair Cases</title>
  <style>
    :root{{--ink:#17211f;--muted:#62706b;--paper:#f4f1e9;--card:#fffefb;--line:#d8d5ca;--green:#1d684f;--blue:#2b5f8d;--orange:#a44e2e}}
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    main{{max-width:1320px;margin:auto;padding:22px 24px 70px}} .hero{{padding:18px 22px;border:1px solid var(--line);border-radius:18px;background:var(--card)}}
    h1{{font-size:clamp(24px,3vw,38px);line-height:1.2;margin:0}} h2,h3,p{{margin-top:0}}
    .filters{{position:sticky;top:0;z-index:8;display:flex;flex-wrap:wrap;gap:8px;margin:18px 0;padding:10px;border:1px solid var(--line);border-radius:14px;background:rgba(244,241,233,.96);backdrop-filter:blur(8px)}}
    button{{border:1px solid #b9c1bd;border-radius:999px;background:white;padding:8px 13px;cursor:pointer}} button.active{{background:var(--ink);color:white;border-color:var(--ink)}}
    .recipe,.special-section{{margin:28px 0;padding:22px;border:1px solid var(--line);border-radius:22px;background:rgba(255,255,255,.55)}} .recipe[hidden]{{display:none}}
    .recipe-head{{display:flex;justify-content:space-between;gap:20px;align-items:end;border-bottom:1px solid var(--line);margin-bottom:16px}} .recipe-head h2{{font-size:25px;margin:4px 0 12px}} .recipe-head p{{max-width:65%;font-weight:650;text-align:right}}
    .case-card{{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--blue);border-radius:18px;padding:18px;margin:16px 0;box-shadow:0 8px 22px rgba(33,42,38,.05)}} .case-card.hybrid_b{{border-left-color:var(--orange)}} .case-card.special{{border-left-color:var(--green)}}
    .case-card>header{{display:flex;justify-content:space-between;gap:18px;align-items:start}} .case-card h3{{margin:2px 0 4px;font-size:21px}} .case-id{{font:700 12px ui-monospace,monospace;color:var(--muted)}} .badges{{display:flex;gap:7px;flex-wrap:wrap;justify-content:end}} .badges b{{font-size:12px;background:#e8ede9;border-radius:999px;padding:5px 9px}} .actions{{font-weight:650;color:#34453f}}
    .video-pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;min-width:0}} .video-panel{{margin:0;min-width:0;background:#101312;border-radius:13px;overflow:hidden}} .video-panel figcaption{{display:flex;justify-content:space-between;padding:9px 11px;color:white}} .video-panel video{{display:block;width:100%;aspect-ratio:16/9;background:#000;object-fit:contain}} .source-line{{color:#d2d9d5;padding:8px 11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px}}
    .section-title{{margin-bottom:14px}} .section-title h2{{font-size:28px;margin:0}}
    @media(max-width:800px){{main{{padding:12px 10px 50px}}.hero{{padding:14px}}.recipe,.special-section{{padding:10px}}.case-card{{padding:10px}}.video-pair{{min-width:720px}}.case-card{{overflow-x:auto}}.recipe-head{{display:block}}.recipe-head p{{max-width:none;text-align:left}}}}
  </style>
</head>
<body>
<main>
  <section class="hero" id="overview">
    <h1>100 个 15 秒全景运镜 Cross-Pair Demo</h1>
  </section>
  <div class="filters" id="combinations"><button class="active" data-filter="all">全部</button>{''.join(f'<button data-filter="{n}">{n} 动作</button>' for n in range(2,7))}</div>
  {''.join(recipe_sections)}
  {special_section}
</main>
<script>
  // 默认展示全部内容；过滤只在用户主动点击后生效，不会偷偷隐藏任何 ref/target。
  document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
    document.querySelectorAll('[data-filter]').forEach(item => item.classList.toggle('active', item === button));
    const wanted = button.dataset.filter;
    document.querySelectorAll('[data-recipe]').forEach(section => {{ section.hidden = wanted !== 'all' && section.dataset.actionCount !== wanted; }});
  }}));

  // 双向同步：用户最后操作的一路成为 leader。播放期间用 requestAnimationFrame
  // 持续检查，漂移超过 0.05 秒就校正 follower；这比只监听 play/pause 更适合
  // 两个编码内容不同、解码耗时也不同的 Cross-Pair。
  document.querySelectorAll('[data-sync-pair]').forEach(pair => {{
    const videos = [...pair.querySelectorAll('video')];
    let syncing = false;
    let leader = videos[0];
    let animationFrame = null;
    const mirror = (source, eventName) => {{
      if (syncing) return; syncing = true;
      videos.filter(video => video !== source).forEach(video => {{
        if (Math.abs(video.currentTime - source.currentTime) > 0.05) video.currentTime = source.currentTime;
        video.playbackRate = source.playbackRate;
        if (eventName === 'play') video.play().catch(() => {{}});
        if (eventName === 'pause') video.pause();
      }});
      requestAnimationFrame(() => syncing = false);
    }};
    const alignWhilePlaying = () => {{
      if (!leader || leader.paused) {{ animationFrame = null; return; }}
      syncing = true;
      videos.filter(video => video !== leader).forEach(video => {{
        if (Math.abs(video.currentTime - leader.currentTime) > 0.05) video.currentTime = leader.currentTime;
        video.playbackRate = leader.playbackRate;
      }});
      syncing = false;
      animationFrame = requestAnimationFrame(alignWhilePlaying);
    }};
    videos.forEach(video => {{
      // 鼠标、触控和键盘操作都可把任意一路切为 leader。
      ['pointerdown','keydown'].forEach(name => video.addEventListener(name, () => leader = video));
      ['play','pause','seeking','ratechange'].forEach(name => video.addEventListener(name, () => {{
        if (syncing) return;
        mirror(video, name);
        if (name === 'play' && animationFrame === null) animationFrame = requestAnimationFrame(alignWhilePlaying);
      }}));
    }});
  }});
</script>
</body></html>
"""


def main() -> None:
    if not PLAN_PATH.exists() or not AUDIT_PATH.exists():
        raise FileNotFoundError("请先运行 pipeline.py audit 与 pipeline.py plan")
    plan = load_json(PLAN_PATH)
    audit = load_json(AUDIT_PATH)
    validation = load_json(VALIDATION_PATH) if VALIDATION_PATH.exists() else None
    SITE_ROOT.mkdir(parents=True, exist_ok=True)
    (SITE_ROOT / "index.html").write_text(build_html(plan, audit, validation), encoding="utf-8")
    (SITE_ROOT / "CASE_MANIFEST.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"网页已生成：{SITE_ROOT / 'index.html'}")


if __name__ == "__main__":
    main()
