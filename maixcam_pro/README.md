# MaixCAM Pro 纯视觉拼图

这个目录是独立的真机版本，不引用原项目的 GUI、MuJoCo、随机生成、机械臂、
串口或动画代码，也不使用深度学习模型。

## 从原项目保留的算法

- `edges` / `candidate_matchings`：枚举边并按相对边长误差筛选切割边；
- `matching_sets`：恢复碎片邻接图，并保证每条边只使用一次；
- `align_edge` / `assemble_from_matches`：反向重合匹配边，传播二维刚体位姿；
- `optimize_pose_graph`：对 3～4 块的闭环端点误差做全局修正；
- `solve`：用重叠、矩形空缺和匹配误差选择最佳拼接，再移动到 A4 下部。

## 两种矩形求解算法

`config.py` 的 `SOLVER_ALGORITHM` 用于选择求解器：

- `1`：原有的邻接图枚举、位姿传播与闭环优化算法，完整保留在
  `puzzle_solver.py`；
- `2`：新增的逐轮轮廓合并算法，实现在 `puzzle_solver_merge.py`，当前默认启用。

算法 2 不枚举“第三刀”或猜测切割过程，而是从检测到的 1～4 个轮廓直接工作：

1. 穷举当前轮廓列表中任意两个轮廓及其边组合；
2. 边长相对误差小于 `EDGE_LENGTH_TOLERANCE`，且至少一组对应端点内角之和
   接近 90° 或 180°时，保留为合并候选；
3. 固定轮廓 A，令 `a=P1-P0`、`b=Q1-Q0`，计算把 `b` 旋转到 `-a` 的
   旋转矩阵 `R`，再使用 `t=P1-RQ0`，得到 `Q''=RQ+t`，即让
   `Q0→P1`、`Q1→P0`；
4. 对齐后取消所有方向相反的公共边，再按剩余边重新追踪外轮廓；这一步也能处理
   最后一块同时与已有组合共享两条边的情况；
5. 对每种合并结果按边长和内角序列去重。一次合并后列表长度必须减少 1，否则
   舍弃该路径；4 块碎片最多连续执行 3 轮；
6. 最终只接受单一四边形、矩形填充率不低于 `MIN_RECTANGLE_FILL` 的候选，
   然后把每次 `R/t` 累乘回每块原始碎片的 3×3 位姿矩阵。

现场如需对比，只改一行即可回退：

```python
SOLVER_ALGORITHM = 1  # 原邻接图算法
```

删除的是仿真数据生成、材质渲染、GUI、动画、MuJoCo 和机械臂控制。当前检测链为：

```text
一次 START → 定位 A4 → 透视校正 → 检测 1～4 块碎片
→ 当前 puzzle_solver.py 几何求解 → 完成指示与位姿输出
```

识别流程参考 `D:\26_new`，但拼图算法仍使用本项目现有 `puzzle_solver.py`。
`workflow.py` 每个相机帧最多推进一个重步骤，三步自动连续执行，中途不需要再次
触摸。普通取景只借用相机 BGR 缓冲显示和绘制少量 UI，不做灰度、Otsu、形态学、
透视、碎片检测或求解，也不再生成四宫格调试画面。

## A4 严格识别

A4 定位按现场讨论的低误检方案实现，但保留本项目已有的 OpenCV/零拷贝架构：

1. 只在屏幕中央 ROI 搜索，候选不得接触 ROI 边缘；
2. 使用 LAB 中性黑阈值，同时限制亮度和 A/B 色度，排除深红、深蓝等物体；
3. 先闭运算填补小亮点，再开运算删除孤立噪声；
4. 只保留面积、A4 长宽比、实心度、对边相似度和角度都合格的凸四边形；
5. 高分辨率 PC 图片通过候选后，仅在四条边附近做局部对比度边缘精修；真机
   640×480 直接使用 LAB 四边形，避免 Canny 占用每帧 CPU；
6. 同一候选连续 3 帧稳定后才锁定，最多搜索 12 帧，避免单帧噪声误触发。

`config.py` 中的 `PAPER_LAB_*` 使用 Sipeed/MaixPy 的 LAB 标度：L 为 0～100，
A/B 为 -128～127；`piece_detector.py` 会转换成 OpenCV 标度。默认引导框按当前
实物布置显示横向 A4，检测本身仍同时支持横向和纵向。

- [Sipeed：LAB 阈值、ROI 与 find_blobs](https://wiki.sipeed.com/maixpy/doc/en/vision/find_blobs.html)
- [Sipeed：OpenCV 与 Maix 图像零拷贝](https://wiki.sipeed.com/maixpy/doc/en/vision/opencv.html)

## 交互与性能设计

屏幕始终只有一个主画面，按下面顺序自动推进：

1. `READY`：全屏实时取景和 A4 对准框，只有一个大号 `START` 按钮；
2. `STEP 1/3`：锁定黑色 A4，显示相机画面和纸张轮廓；
3. `STEP 2/3`：显示透视校正后的单张 A4 与碎片编号；
4. `DONE`：显示绿色拼图结果、填充率和总耗时，位姿完整值输出到终端；
5. 仅失败时显示 `RETRY`，完成后可点 `NEW RUN` 回到实时对准页。

性能路径参考 Sipeed 官方 MaixPy 文档：MaixCAM Pro 使用屏幕原生
`640×480`，相机请求 `60 FPS` 和 3 个缓冲；实时帧通过
`image.image2cv(..., copy=False)` 借用后直接 `display.show(maix_frame)`，避免
每帧 OpenCV/Maix 来回转换和大图拼接。完成页保持静态，不继续无意义地取相机和
刷新屏幕。屏幕上的 FPS 使用官方 `maix.time.fps_start()` / `time.fps()` API。

- [Sipeed：MaixCAM 摄像头与帧率](https://wiki.sipeed.com/maixpy/doc/zh/vision/camera.html)
- [Sipeed：OpenCV 零拷贝转换注意事项](https://wiki.sipeed.com/maixpy/doc/en/vision/opencv.html)
- [Sipeed：触摸坐标与显示图像尺寸](https://wiki.sipeed.com/maixpy/doc/zh/vision/touchscreen.html)
- [Sipeed：MaixPy 应用界面与大按钮建议](https://wiki.sipeed.com/maixpy/doc/zh/basic/app.html)

## PC 图片测试

在仓库根目录执行：

```powershell
.venv\Scripts\python.exe maixcam_pro\pc_test.py C:\path\to\image.png
.venv\Scripts\python.exe maixcam_pro\pc_test.py C:\path\to\image.png --algorithm 1
.venv\Scripts\python.exe maixcam_pro\pc_test.py C:\path\to\image.png --algorithm 2
```

输出写入 `output/maixcam_pro_pc/`；若文件已存在会自动加序号，不覆盖旧结果。
`detected.png` 与 `solved.png` 分别是设备第 2、3 步的单页画面，另有
`piece_binary.png` 和完整 `motions.json`。
现场若检测不稳定，先使用 MaixCAM 自带 Find Blobs 工具读取黑纸 LAB 值，再优先调整
`PAPER_LAB_L_MAX`、`PAPER_LAB_A_MIN/MAX`、`PAPER_LAB_B_MIN/MAX`、
`PAPER_MIN_AREA_RATIO` 和 `PAPER_MIN_FILL_RATIO`。碎片阶段再调整
`PIECE_GRAY_MIN` 与 `PIECE_MIN_AREA_RATIO`。

## MaixCAM Pro 运行

1. 在 MaixVision 中打开整个 `maixcam_pro` 文件夹，不要只打开单个 `main.py`；
2. 连接 MaixCAM Pro，直接运行根目录的 `main.py`；
3. 相机需完整看到黑色 A4，黑纸四周保留明亮背景，碎片不能接触或遮挡；
4. 按赛题流程移除摄像头遮挡，同时只点一次 `START`；
5. 屏幕会自动逐步显示 A4 定位、碎片识别、拼图求解和完成结果；
6. 中途不需要人工操作；失败时检查摆放后点 `RETRY`；
7. MaixVision 终端输出真实设备 FPS、阶段总耗时，以及每块的
   `rotation`、`dx`、`dy`、`distance`。

位姿采用透视校正后的 A4 像素坐标：x 向右、y 向下；正旋转角在屏幕上表现
为顺时针。默认 A4 平面为 420×594 px，即 2 px/mm；修改尺寸会同时改变输出像素尺度。
机械结构若需要毫米量，仍需另做相机标定；本目录只输出题目要求的像素量。

如果检测已经得到 1～4 个正确多边形，但提示“未找到满足边长和邻接关系”，说明
输入碎片的边长无法按容差组成原项目规定的闭合邻接图。此时不应盲目增大容差，
应先检查碎片是否确实来自同一矩形，以及相机透视是否已足够小。

PC 测试只能验证算法、界面尺寸和阶段耗时，不能代表 MaixCAM Pro 真机 FPS。
真机验收请以屏幕 FPS 和 MaixVision 终端每 60 帧打印的日志为准。
