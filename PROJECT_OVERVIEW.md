# QFlux Video Mesh Flow System

## 项目定位

本项目用于将单目视频转换为稳定的 3D 人体 Mesh / USD 输出，用于 Blender、点云、视频参考、位移或后续 3D 工作流。

当前系统的核心目标是：

- 从单个视频生成稳定人体结构 Mesh。
- 基于 4DHumans / WHAM 输出人体结构缓存。
- 生成时序一致的 Body / Dense Body Mesh。
- 基于 Human Parsing 分割缓存生成衣服 / 头发的可见几何增强层。
- 输出分层 USD，方便在 Blender 中单独检查 Body、Garment、Hair。

本项目不是完整真实衣服 / 头发 3D 重建系统。当前 Garment / Hair 的目标是：

> 基于人体 Mesh + 2D 分割结果，生成稳定、可见、可分层检查的壳层与轮廓近似几何。

也就是说，它是用于视觉参考和 3D 流程衔接的近似几何，不是物理布料、真实发丝或完整服装拓扑。

---

## 当前主流程

### 1. 输入与预处理

用户导入主视频或图片序列。

统一预处理规则：

- 如果输入存在 Alpha，透明区域自动合成黑色背景。
- 如果输入没有 Alpha，直接使用原图。
- Alpha 不作为 UI 主判断条件，不再依赖“检测成功 / 检测失败”决定流程。
- 4DHumans、WHAM、Human Parsing、预览与导出调试图必须尽量使用同一套预处理结果。

目的：

- 避免不同模块看到不同输入。
- 避免透明背景导致人体结构模型或分割模型误判。
- 避免 Alpha 检测失败后静默进入错误流程。

---

### 2. 人体结构生成

人体结构由 4DHumans / WHAM 生成。

输出结构缓存包含：

- vertices
- faces
- joints
- camera / weak camera 信息
- confidence / track 信息
- frame index 对齐信息

WHAM 不能直接读取原视频，应使用预处理后的黑底视频或预处理帧序列。

目的：

- 保证 WHAM 与 4DHumans 使用一致输入。
- 避免 Alpha、裁切、补底等预处理只对部分模型生效。

---

### 3. 时序稳定

人体结构生成后进行稳定处理。

主要包括：

- root alignment
- temporal smoothing
- pose stabilization
- fixed topology frame sequence

默认目标是获得稳定可看的原地人体 Mesh。

后续如果需要保留人物在场景中的真实位移，应提供两种模式：

- 稳定原地模型：锁定 root，适合动作和形体检查。
- 保留真实位移：保留 global / root trajectory，适合导入 Blender 场景。

---

### 4. Human Parsing 分割缓存

衣服 / 头发几何增强依赖 Human Parsing 分割缓存。

当前规则：

- 有有效分割缓存：才允许生成 Garment / Hair 层。
- 无有效分割缓存：明确进入 Body Only 模式。
- 不再用身高规则假装生成衣服 / 头发。
- 不再在缺失分割时输出误导性的 mesh_garment.usda / mesh_hair.usda。

Human Parsing 主要用于生成：

- garment mask
- hair mask
- region weights

分割标签需要合并成两类主区域：

- Garment：上衣、裤子、裙子、外套、连衣裙等。
- Hair：头发区域。

鞋子、帽子、围巾等标签不应无条件混入主衣服 / 头发逻辑，应按后续需求单独处理或忽略。

---

### 5. Region Weights 缓存

分割结果不会直接等于 3D 几何。

流程应先生成 region_weights.npz：

- garment_weight
- hair_weight
- body_weight
- 对应 frame / mesh 拓扑信息

预览和导出都应优先读取同一份 region_weights.npz。

目的：

- 避免预览和导出使用不同 fallback。
- 避免同一个视频在预览里看起来有效，导出后结果不同。
- 让 Garment / Hair 是否生效可检查、可复现。

---

### 6. Mesh 生成

#### Body Mesh

Body Mesh 是主结构表达。

来源：

- 4DHumans / WHAM 输出的稳定人体结构。
- 可经过 dense mesh 细分。

要求：

- 拓扑尽量固定。
- 不逐帧 remesh。
- 优先保证时序稳定。

#### Garment Shell

Garment Shell 是衣服区域壳层，不是真实服装拓扑。

当前生成方式：

- 根据 garment_weight 选择身体表面的衣服区域。
- 沿法线方向生成外扩壳层。
- 对边界生成 silhouette / side wall，让衣服层在 Blender 中更容易看见。

限制：

- 无法生成真实宽松衣服内部结构。
- 无法准确还原复杂裙摆、飘带、外套开口等真实拓扑。
- 主要用于让衣服区域成为独立、可见、可检查的几何层。

#### Hair Shell

Hair Shell 是头发区域近似层，不是真实发丝系统。

当前生成方式：

- 根据 hair_weight 选择头部附近区域。
- 沿法线生成头发壳层。
- 根据 mask 边界生成 silhouette / side wall。

限制：

- 长发、散发、遮挡严重的头发无法完全真实还原。
- 当前重点是让头发区域在 Blender 中有独立层和可见轮廓。

---

### 7. USD 输出结构

当前推荐输出：

```text
mesh_body.usda
mesh_garment.usda
mesh_hair.usda
mesh_combined.usda
```

#### mesh_body.usda

包含：

```text
/Body
```

#### mesh_garment.usda

有有效分割并启用衣服层时输出：

```text
/GarmentShell
/GarmentSilhouette
```

#### mesh_hair.usda

有有效分割并启用头发层时输出：

```text
/HairShell
/HairSilhouette
```

#### mesh_combined.usda

Combined 必须是分层结构，不应再是单个混合 Mesh。

有完整分割并启用衣服 / 头发时，结构为：

```text
/Body
/GarmentShell
/GarmentSilhouette
/HairShell
/HairSilhouette
```

如果只启用衣服：

```text
/Body
/GarmentShell
/GarmentSilhouette
```

如果只启用头发：

```text
/Body
/HairShell
/HairSilhouette
```

如果没有有效分割：

```text
/Body
```

要求：

- Body / Garment / Hair 使用不同 displayColor。
- 导入 Blender 后应能单独选择、隐藏和检查不同层。
- UI 开关关闭某层时，导出文件中不应写入该层。

---

### 8. 预览系统

预览的目标是“可信判断”，不是只追求好看。

预览应优先显示：

- Body Mesh
- Garment Shell
- Garment Silhouette
- Hair Shell
- Hair Silhouette
- Combined 分层结果

预览原则：

- 不再用自动猜最大轴的方式决定主视角。
- 使用固定正面 / 侧面 / 近似原视频视角。
- Body、Garment、Hair 使用不同颜色。
- 没有分割缓存时明确显示 Body Only。
- 预览与导出必须尽量使用同一份 region_weights。

---

## 当前已修正的关键问题

### Alpha 处理

旧逻辑：

- 检测 Alpha。
- 根据检测结果决定 UI 状态。
- 部分视频 Alpha 会被 OpenCV 丢失。
- 函数签名和返回值存在不一致。

当前逻辑：

- 有 Alpha 就自动黑底合成。
- 没有 Alpha 就原样输入。
- Alpha 不再作为主流程开关。

---

### WHAM 输入一致性

旧逻辑：

- WHAM 可能直接读取原始视频。
- 4DHumans / 分割 / 预览可能使用另一套预处理结果。

当前逻辑：

- WHAM 应读取预处理后的黑底视频或帧序列。
- 保证结构生成链路输入一致。

---

### Garment / Hair 假输出

旧逻辑：

- 没有分割缓存时，可能用身体高度规则生成假衣服 / 假头发。
- mesh_garment.usda / mesh_hair.usda 看似存在，但实际只是身体区域。

当前逻辑：

- 没有有效 Human Parsing：Body Only。
- 不再输出误导性的 Garment / Hair。
- 有分割缓存并启用开关时，才输出对应层。

---

### Combined 分层

旧逻辑：

- mesh_combined.usda 可能只是单个外扩后的 Body Mesh。
- Blender 导入后看不出衣服 / 头发。

当前逻辑：

- Combined 使用多个 prim 分层。
- Body / Garment / Hair 可单独检查。
- Garment / Hair 有 shell 和 silhouette 结构。

---

### 点云密度自定义

旧逻辑：

- UI 有“自定义”选项，但没有完整 spinbox 和返回值处理。
- 可能返回 None 导致崩溃。

当前逻辑：

- 短期只保留低 / 中 / 高。
- 不再暴露未完成的自定义入口。

---

## 缓存与清理

为了避免旧输出误导，更新代码后建议清理旧缓存和旧 USD。

建议清理对象：

```text
__pycache__
*.pyc
*.pyo
data/cache
项目 cache
mesh_body.usda
mesh_garment.usda
mesh_hair.usda
mesh_combined.usda
region_weights.npz
wham_preprocessed_black.mp4
旧 mesh output
旧 pointcloud output
```

不能清理：

```text
app 源码
models
external_repos
.venv
输入视频
4DHumans / WHAM / FASHN 权重
```

清理后推荐顺序：

```text
重启 GUI
重新生成结构缓存
重新生成分割缓存
重新导出 USD
重新导入 Blender 检查 mesh_combined.usda
```

---

## 核心设计原则

### 1. Mesh 优先

Mesh 是主表达。点云是可选输出，不应替代 Mesh 预览。

### 2. 分割是约束，不是事实

Human Parsing 只提供 2D 区域参考。

它不能直接代表真实 3D 几何。

### 3. 无分割不假装成功

没有有效分割时，只输出 Body。

不生成假 Garment / Hair。

### 4. 预览和导出必须一致

预览和导出应共用 region_weights。

不能预览一套逻辑，导出另一套逻辑。

### 5. Combined 必须分层

Combined 不应只是一个单 Mesh。

Blender 中应能看到并单独检查：

```text
Body
GarmentShell
GarmentSilhouette
HairShell
HairSilhouette
```

### 6. UI 不显示未接通功能

如果某功能只是 stub、return None 或旧流程残留，不应显示给用户。

---

## 坐标系规范

App 内部假定 Y-Up：

- Y 轴朝上。
- 数值越大越靠近头顶。
- 高度相关启发式都基于 Y-Up。

外部模型如果使用相机坐标系或 Y-down，需要在导入时转换：

- 翻转必要轴向。
- 保持右手坐标系。
- 保证导入后人体不是倒置或镜像。

---

## 当前限制

当前系统仍有这些限制：

- Garment / Hair 不是完整真实 3D 重建。
- 复杂宽松衣服、长发、遮挡、裙摆无法完全准确。
- 2D mask 到 3D mesh 的映射仍依赖近似投影或权重映射。
- 没有高质量分割缓存时，只能输出 Body。
- GUI 完整启动、Blender 导入、真实长视频全流程仍需要本地环境验证。

---

## 后续可升级方向

### 1. 相机投影替代 bbox 映射

使用 4DHumans / WHAM 的相机参数进行 3D vertex 到 2D image 的投影。

目标：

- 减少侧身、转身时 mask 映射错位。
- 提升衣服 / 头发权重准确度。

### 2. 逐帧或关键帧 region weights

当前可先使用缓存权重。

后续可升级为：

- 快速模式：第一帧权重。
- 平衡模式：每 N 帧计算一次，时间平滑。
- 高质量模式：逐帧计算。

### 3. 更强的 Garment Silhouette Mesh

后续可根据 2D 衣服轮廓生成更明显的外轮廓片。

目标：

- 改善宽松衣服、裙摆、外套轮廓。
- 让衣服不只贴在身体表面。

### 4. 更强的 Hair Volume

后续可基于头部 anchor + hair mask 生成更稳定的头发体积层。

目标：

- 改善长发、侧发、后脑轮廓。
- 减少头发完全贴头皮的问题。

### 5. 输出质量检查

导出前自动检查：

- Garment 顶点数是否过少。
- Hair 顶点数是否过少。
- offset 是否全为 0。
- 是否存在 NaN / inf。
- 是否和 Body 完全重合。
- mask 覆盖率是否异常。

失败时不假装成功，直接提示 Body Only 或对应层失败。
