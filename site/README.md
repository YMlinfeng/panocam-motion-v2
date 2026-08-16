# 静态网页目录

- `index.html`：由 `scripts/build_site.py` 生成的入口。
- `media/cases/<case_id>/ref.mp4`：左侧参考场景。
- `media/cases/<case_id>/target.mp4`：右侧目标场景。
- `media/cases/<case_id>/ref.jpg` / `target.jpg`：网页首屏预览图；播放后仍使用高清 MP4。
- `media/cases/<case_id>/metadata.json`：该 case 的动作、源窗口、编码参数、轨迹安全报告和 SHA-256。
- `CASE_MANIFEST.json`：全部 100 个 case 的总清单。

网页默认展示全部 100 个 case；每张卡片始终保留左右视频，不会因响应式布局隐藏其中一路。
