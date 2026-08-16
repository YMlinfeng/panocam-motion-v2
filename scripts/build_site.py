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
        mode_zh = "顺序组合" if case["mode"] == "sequential" else "叠加组合"
        actions = " + ".join(ACTION_ZH[item["type"]] for item in case["actions"])
        title = f"{mode_zh} · {case['action_count']} 动作"
        kind_class = case["mode"]
    else:
        title = "小行星（不组合）" if case["kind"] == "tiny_planet" else "兔子洞（不组合）"
        actions = "特殊球极投影；不叠加 A 类动作"
        kind_class = "special"
    return f"""
    <article class="case-card {kind_class}" data-mode="{escape(case['mode'])}" data-actions="{case['action_count']}">
      <header>
        <div><span class="case-id">{escape(case['id'])}</span><h3>{title}</h3></div>
        <div class="badges"><b>15.00s</b><b>{width}×{height}</b><b>{width * height:,} px</b></div>
      </header>
      <p class="actions">{escape(actions)}</p>
      <div class="video-pair" data-sync-pair>
        {video_panel(case, 'ref')}
        {video_panel(case, 'target')}
      </div>
      <p class="sync-note">左右使用同一逐帧轨迹；任意一路播放、暂停、拖动或调速，另一路会同步。</p>
    </article>"""


def source_table(audit: dict[str, Any]) -> str:
    rows = []
    for source in audit["sources"]:
        metadata = source.get("metadata", {})
        usable = source.get("usable_windows", [])
        if usable:
            dynamic = sum(item["dynamic_score"] for item in usable) / len(usable)
            global_flow = sum(item["global_flow_px"] for item in usable) / len(usable)
            metric = f"dynamic {dynamic:.4f} · global {global_flow:.3f}px"
        else:
            metric = "无合格 15 秒窗口"
        status = source["final_status"]
        rows.append(f"""
          <tr class="{status}">
            <td><b>{escape(source['dataset'])}</b></td>
            <td title="{escape(source['file'])}">{escape(source['file'])}</td>
            <td>{metadata.get('width', '—')}×{metadata.get('height', '—')} · {float(metadata.get('fps', 0)):.2f}fps</td>
            <td>{len(usable)}</td>
            <td>{escape(metric)}</td>
            <td><b>{'保留' if status == 'pass' else '剔除'}</b><small>{escape(source['final_reason'])}</small></td>
          </tr>""")
    return "".join(rows)


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
        cases.sort(key=lambda item: 0 if item["mode"] == "sequential" else 1)
        action_text = " + ".join(ACTION_ZH[item["type"]] for item in cases[0]["actions"])
        recipe_title = cases[0].get("display_name") or recipe_id
        origin_badge = " · LEGACY RETAINED" if cases[0].get("origin") == "legacy_retained_exact_A_only" else ""
        recipe_sections.append(f"""
        <section class="recipe" data-recipe data-action-count="{cases[0]['action_count']}">
          <div class="recipe-head">
            <div><span>COMBINATION FAMILY{origin_badge}</span><h2>{escape(recipe_title)}</h2></div>
            <p>{escape(action_text)}</p>
          </div>
          {''.join(case_card(case) for case in cases)}
        </section>""")

    special_section = f"""
      <section class="special-section" id="specials">
        <div class="section-title"><span>STANDALONE SPECIALS</span><h2>小行星 5 条 + 兔子洞 5 条</h2>
        <p>按要求不与任何 A 类动作组合；两种特殊投影合计 10 个 case。</p></div>
        {''.join(case_card(case) for case in specials)}
      </section>"""

    valid_badge = "全量验证通过" if validation and validation.get("passed") else "等待全量验证"
    total_bytes = int(validation.get("total_video_bytes", 0)) if validation else 0
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Panocam Motion V2 · 100 Cross-Pair Cases</title>
  <style>
    :root{{--ink:#17211f;--muted:#62706b;--paper:#f4f1e9;--card:#fffefb;--line:#d8d5ca;--green:#1d684f;--blue:#2b5f8d;--orange:#a44e2e}}
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    nav{{position:sticky;top:0;z-index:10;display:flex;gap:14px;align-items:center;padding:12px max(24px,calc((100vw - 1320px)/2));background:rgba(23,33,31,.96);color:white;overflow-x:auto;white-space:nowrap}}
    nav b{{margin-right:auto}} nav a{{color:white;text-decoration:none;opacity:.85}} nav a:hover{{opacity:1}}
    main{{max-width:1320px;margin:auto;padding:28px 24px 80px}} .hero{{padding:36px;border:1px solid var(--line);border-radius:24px;background:linear-gradient(135deg,#fffefb,#e6eee7)}}
    .eyebrow,.section-title>span,.recipe-head span{{font-size:12px;letter-spacing:.14em;font-weight:800;color:var(--green)}} h1{{font-size:clamp(36px,6vw,72px);line-height:1;margin:10px 0 18px}} h2,h3,p{{margin-top:0}}
    .hero p{{max-width:950px;font-size:18px}} .stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:24px}} .stat{{padding:16px;border-radius:14px;background:white;border:1px solid var(--line)}} .stat b{{display:block;font-size:25px}}
    .rules{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0 0;padding:0;list-style:none}} .rules li{{background:#17211f;color:white;border-radius:14px;padding:16px}}
    .filters{{position:sticky;top:48px;z-index:8;display:flex;flex-wrap:wrap;gap:8px;margin:28px 0;padding:14px;border:1px solid var(--line);border-radius:16px;background:rgba(244,241,233,.95);backdrop-filter:blur(8px)}}
    button{{border:1px solid #b9c1bd;border-radius:999px;background:white;padding:8px 13px;cursor:pointer}} button.active{{background:var(--ink);color:white;border-color:var(--ink)}}
    .recipe,.special-section,.dataset{{margin:28px 0;padding:22px;border:1px solid var(--line);border-radius:22px;background:rgba(255,255,255,.55)}} .recipe[hidden]{{display:none}}
    .recipe-head{{display:flex;justify-content:space-between;gap:20px;align-items:end;border-bottom:1px solid var(--line);margin-bottom:16px}} .recipe-head h2{{font-size:25px;margin:4px 0 12px}} .recipe-head p{{max-width:65%;font-weight:650;text-align:right}}
    .case-card{{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--blue);border-radius:18px;padding:18px;margin:16px 0;box-shadow:0 8px 22px rgba(33,42,38,.05)}} .case-card.simultaneous{{border-left-color:var(--orange)}} .case-card.special{{border-left-color:var(--green)}}
    .case-card>header{{display:flex;justify-content:space-between;gap:18px;align-items:start}} .case-card h3{{margin:2px 0 4px;font-size:21px}} .case-id{{font:700 12px ui-monospace,monospace;color:var(--muted)}} .badges{{display:flex;gap:7px;flex-wrap:wrap;justify-content:end}} .badges b{{font-size:12px;background:#e8ede9;border-radius:999px;padding:5px 9px}} .actions{{font-weight:650;color:#34453f}}
    .video-pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;min-width:0}} .video-panel{{margin:0;min-width:0;background:#101312;border-radius:13px;overflow:hidden}} .video-panel figcaption{{display:flex;justify-content:space-between;padding:9px 11px;color:white}} .video-panel video{{display:block;width:100%;aspect-ratio:16/9;background:#000;object-fit:contain}} .source-line{{color:#d2d9d5;padding:8px 11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px}}
    .sync-note{{margin:10px 0 0;color:var(--muted);font-size:12px}} .section-title{{margin-bottom:20px}} .section-title h2{{font-size:32px;margin:5px 0}}
    .dataset{{overflow:auto}} table{{width:100%;border-collapse:collapse;min-width:1050px;background:white}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}} th{{background:#17211f;color:white}} td:nth-child(2){{max-width:330px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}} td small{{display:block;color:var(--muted)}} tr.reject{{opacity:.55}} tr.pass td:last-child b{{color:var(--green)}}
    footer{{max-width:1320px;margin:auto;padding:0 24px 50px;color:var(--muted)}}
    @media(max-width:800px){{main{{padding:16px 10px 60px}}.hero{{padding:22px}}.stats{{grid-template-columns:repeat(2,1fr)}}.rules{{grid-template-columns:1fr}}.recipe,.special-section,.dataset{{padding:10px}}.case-card{{padding:10px}}.video-pair{{min-width:720px}}.case-card{{overflow-x:auto}}.recipe-head{{display:block}}.recipe-head p{{max-width:none;text-align:left}}}}
  </style>
</head>
<body>
<nav><b>Panocam Motion V2</b><a href="#overview">总览</a><a href="#combinations">90 个组合</a><a href="#specials">10 个特殊投影</a><a href="#sources">素材筛选</a></nav>
<main>
  <section class="hero" id="overview">
    <span class="eyebrow">PANORAMIC VIRTUAL CAMERA · VERSION 2</span>
    <h1>100 个 15 秒<br>Cross-Pair 高清示例</h1>
    <p>只保留旋转/小幅变焦的组合运镜，并使用动态画面、静止全景机位素材。普通组合在渲染前做 ERP 接缝安全检查；ref 与 target 的轨迹、时长、分辨率逐帧一致。</p>
    <div class="stats"><div class="stat"><b>100</b>cases</div><div class="stat"><b>200</b>左右 MP4</div><div class="stat"><b>15.00s</b>每条</div><div class="stat"><b>≤ 960²</b>max pixels</div><div class="stat"><b>{valid_badge}</b>{total_bytes/1024**2:.1f} MiB</div></div>
    <ul class="rules"><li><b>已删除</b><br>B_transl、C_optics、全部代理及含代理组合</li><li><b>组合结构</b><br>45 个动作配方 × 顺序/叠加 = 90 case</li><li><b>特殊投影</b><br>小行星 5 + 兔子洞 5；均不参与组合</li></ul>
  </section>
  <div class="filters" id="combinations"><b>显示动作数：</b><button class="active" data-filter="all">全部 45 个配方</button>{''.join(f'<button data-filter="{n}">{n} 动作</button>' for n in range(2,7))}</div>
  {''.join(recipe_sections)}
  {special_section}
  <section class="dataset" id="sources"><div class="section-title"><span>SOURCE SCREENING LEDGER</span><h2>固定机位动态素材筛选</h2><p>算法指标与人工内容复核同时通过才会进入生成池；表中“可用窗口”都是完整 15 秒。</p></div>
    <table><thead><tr><th>数据集</th><th>文件</th><th>原始规格</th><th>可用窗口</th><th>窗口指标均值</th><th>决定</th></tr></thead><tbody>{source_table(audit)}</tbody></table>
  </section>
</main>
<footer>代码、两份说明文档、逐 case 参数、素材筛选台账与验证报告均随公开仓库发布。网页直接播放生产 MP4，不使用 GIF，也不做浏览器端二次压缩。</footer>
<script>
  // 默认展示全部内容；过滤只在用户主动点击后生效，不会偷偷隐藏任何 ref/target。
  document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
    document.querySelectorAll('[data-filter]').forEach(item => item.classList.toggle('active', item === button));
    const wanted = button.dataset.filter;
    document.querySelectorAll('[data-recipe]').forEach(section => {{ section.hidden = wanted !== 'all' && section.dataset.actionCount !== wanted; }});
  }}));

  // 双向同步：播放、暂停、seek、倍速都镜像。0.08 秒容差避免浏览器解码抖动造成死循环。
  document.querySelectorAll('[data-sync-pair]').forEach(pair => {{
    const videos = [...pair.querySelectorAll('video')];
    let syncing = false;
    const mirror = (source, eventName) => {{
      if (syncing) return; syncing = true;
      videos.filter(video => video !== source).forEach(video => {{
        if (Math.abs(video.currentTime - source.currentTime) > 0.08) video.currentTime = source.currentTime;
        video.playbackRate = source.playbackRate;
        if (eventName === 'play') video.play().catch(() => {{}});
        if (eventName === 'pause') video.pause();
      }});
      requestAnimationFrame(() => syncing = false);
    }};
    videos.forEach(video => ['play','pause','seeking','ratechange'].forEach(name => video.addEventListener(name, () => mirror(video, name))));
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
