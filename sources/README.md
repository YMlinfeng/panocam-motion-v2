# 原始素材说明

`sources/raw/` 被 `.gitignore` 排除，因为完整原始数据体积很大，而且不同数据集许可不同。公开仓库保留：

- `config/pipeline.yaml`：来源、许可和人工复核决定；
- `output/SOURCE_AUDIT.json`：逐窗口动态分数、全局运动分数和最终去留；
- `output/CASE_PLAN.json`：每条 demo 实际使用的源文件与时间窗。

本机默认从相邻旧项目已经下载的只读目录读取素材；也可以把数据放到任意目录，然后使用：

```bash
local/.venv/bin/python scripts/pipeline.py audit \
  --source-root-360x /你的/360x/hr_samples \
  --source-root-commons /你的/Wikimedia/videos
```

