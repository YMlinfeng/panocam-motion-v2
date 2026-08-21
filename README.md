# Panocam Motion V2

Panocam Motion V2 是一个从零重写的全景视频虚拟运镜项目。当前版本只生产两类内容：

1. 90 个由 2～6 种旋转/变焦基础动作组成的连续混合时序：45 个动作配方各生成 A/B 两个独立变体；动作数≥3时，至少3种运镜共享2.5～5秒的同时叠加区间。
2. 10 个动态特殊投影：5 个小行星、5 个兔子洞；每条持续顺/逆时针滚转，并叠加轻微呼吸 Zoom。

每个 case 都包含左右两条 Cross-Pair 视频。两条视频使用完全相同的逐帧轨迹、时长和输出分辨率，只更换源场景；每条严格 15 秒，输出像素不超过 `960×960 = 921,600`。

本版本明确不包含：B 类位移运镜、C 类光学/畸变类目、任何代理几何、单独 A 类展示、含代理的组合运镜、纯顺序或纯叠加 case。

速度、幅度、动作起止、持续时间和曲线均随机化，并人工加入“快速为主、范围较大、同时组合明显”的偏置。全100个 case 中含 roll 的总量控制为18%；360+x 人脸模糊素材和3D/动画素材只进入 REF，TARGET 只使用真实场景。

## 在线访问

- 公开 GitHub 仓库：<https://github.com/YMlinfeng/panocam-motion-v2>
- 100-case 高清网页：<https://ymlinfeng.github.io/panocam-motion-v2/>
- 20-case 真实位移实验：<https://ymlinfeng.github.io/panocam-motion-v2/translation/>

网页发布分支为 `gh-pages`，主分支 `main` 保存完整代码、两份文档、生产清单、验证台账和同一份网页资源。

## 真实位移补充实验

`scripts/translation_pairs.py` 使用移动全景相机视频本身提供真实空间位移：每个
ref/target 都来自同一源视频中尽量远的两个时间窗口。独立演示页包含 10 个原生
位移 case 和 10 个“真实位移 + 两侧同步虚拟旋转”case，每侧原速 10 秒、20fps，
像素数不超过 960×960。该方法适合静态或近静态环境；若场景内有显著动态物体，
两个时间窗口不会保持逐帧语义同步，需要在全量生产前单独筛查。

```bash
local/.venv/bin/python scripts/translation_pairs.py all \
  --source-dir /Users/bytedance/Downloads/数据集 --workers 2
```

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

两者是彼此独立的 Git 仓库。新版不会读取旧版代码或旧 demo；本机只读引用旧目录中的 360+x 5K 数据，Commons/NASA 视频位于 `local/sources/commons`，真实静态全景位于 `local/sources/commons_still`。公开仓库配置提供了改成本地其他路径的方法。
