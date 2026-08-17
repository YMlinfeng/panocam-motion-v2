# 原始素材说明

`sources/raw/` 被 `.gitignore` 排除，因为完整原始数据体积很大，而且不同数据集许可不同。公开仓库保留：

- `config/pipeline.yaml`：来源、许可和人工复核决定；
- `output/SOURCE_AUDIT.json`：逐窗口动态分数、全局运动分数和最终去留；
- `output/CASE_PLAN.json`：每条 demo 实际使用的源文件与时间窗。

当前角色规则：360+x 官方人脸模糊素材以及3D/动画素材只进入 REF；TARGET 只使用
真实实拍视频或真实静态360全景。长视频最多取8个不同15秒窗口，并用最低使用
次数优先的分配器减少重复。

本机默认只读引用相邻旧项目的 360+x 5K 数据；Commons/NASA 文件位于当前项目
`local/sources/commons`，真实静态全景位于 `local/sources/commons_still`。也可以把
数据放到任意目录，然后使用：

```bash
local/.venv/bin/python scripts/pipeline.py audit \
  --source-root-360x /你的/360x/hr_samples \
  --source-root-commons /你的/Wikimedia/videos
```
