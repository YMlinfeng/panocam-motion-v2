# Panocam Motion V2

Panocam Motion V2 是一个从零重写的全景视频虚拟运镜项目。它只生产两类内容：

1. 90 个由 2～6 种旋转/变焦基础动作组成的组合运镜；45 个动作配方各生成一条顺序组合和一条叠加组合。
2. 10 个不参与组合的特殊投影：5 个小行星、5 个兔子洞。

每个 case 都包含左右两条 Cross-Pair 视频。两条视频使用完全相同的逐帧轨迹、时长和输出分辨率，只更换源场景；每条严格 15 秒，输出像素不超过 `960×960 = 921,600`。

本版本明确不包含：B 类位移运镜、C 类光学/畸变类目、任何代理几何、单独 A 类展示、含代理的组合运镜。

旧版仍符合新约束的两个纯 A 组合被保留为 `recipe_41`（Vertigo Roll）和 `recipe_42`（Pan → Zoom → Tilt）；其余 recipe 用固定随机种子扩充。

## 一键开始

```bash
cd /Users/bytedance/Downloads/GitHubProjects/panocam-motion-v2
python3 -m venv local/.venv
local/.venv/bin/python -m pip install -r requirements.txt

# 1. 筛选原始全景素材
local/.venv/bin/python scripts/pipeline.py audit

# 2. 生成固定随机种子的 100-case 计划
local/.venv/bin/python scripts/pipeline.py plan

# 3. 先渲染 2 条做冒烟检查，再渲染全部
local/.venv/bin/python scripts/pipeline.py render --limit 2
local/.venv/bin/python scripts/pipeline.py render --workers 2

# 4. 严格验证并生成网页
local/.venv/bin/python scripts/pipeline.py validate
local/.venv/bin/python scripts/build_site.py
```

所有参数、阈值和调节方法见 [方案详解](docs/01_方案详解.md) 与 [生产手册](docs/02_生产手册.md)。

## 新旧项目位置

- 新版：`/Users/bytedance/Downloads/GitHubProjects/panocam-motion-v2`
- 旧版历史材料：`/Users/bytedance/Downloads/GitHubProjects/panocam`

两者是彼此独立的 Git 仓库。新版不会修改、复制或隐式依赖旧版代码；本机默认会从旧版已经下载的数据目录读取原始 360° 素材，公开仓库中的配置也提供了改成本地其他路径的方法。

## 许可

- 代码与文档：MIT。
- 360+x 来源及其派生 demo：CC BY-NC-SA 4.0，版权与署名归原数据集及原作者。
- Wikimedia Commons 来源及其派生 demo：按逐条 `SOURCE_AUDIT.json` 中记录的原始许可执行。
