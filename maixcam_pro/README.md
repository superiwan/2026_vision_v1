# MaixCAM Pro 纯视觉拼图

这个目录是独立的真机版本，不引用原项目的 GUI、MuJoCo、随机生成、机械臂、
串口或动画代码，也不使用深度学习模型。

## 从原项目保留的算法

- `edges` / `candidate_matchings`：枚举边并按相对边长误差筛选切割边；
- `matching_sets`：恢复碎片邻接图，并保证每条边只使用一次；
- `align_edge` / `assemble_from_matches`：反向重合匹配边，传播二维刚体位姿；
- `optimize_pose_graph`：对 3～4 块的闭环端点误差做全局修正；
- `solve`：用重叠、矩形空缺和匹配误差选择最佳拼接，再移动到 A4 下部。

删除的是仿真数据生成、材质渲染、GUI、动画、MuJoCo 和机械臂控制。检测部分改为
“最大黑色 A4 外轮廓 → 纸内非黑色阈值 → 外轮廓 → `approxPolyDP` 3～5 点”。

## PC 图片测试

在仓库根目录执行：

```powershell
.venv\Scripts\python.exe maixcam_pro\pc_test.py C:\path\to\image.png
```

输出写入 `output/maixcam_pro_pc/`；若文件已存在会自动加序号，不覆盖旧结果。
现场若检测不稳定，优先调整 `config.py` 的 `PAPER_GRAY_MAX`、
`PIECE_GRAY_MIN` 和 `PIECE_MIN_AREA_RATIO`。

## MaixCAM Pro 运行

1. 在 MaixVision 中打开整个 `maixcam_pro` 文件夹，不要只打开单个 `main.py`；
2. 连接 MaixCAM Pro，直接运行根目录的 `main.py`；
3. 相机需完整看到黑色 A4，碎片不能接触或遮挡；
4. 屏幕显示当前轮廓、P0～P3、绿色目标矩形，以及旋转角和像素移动量；
5. MaixVision 终端也会输出每块的 `rotation`、`dx`、`dy`、`distance`。

像素坐标采用相机画面坐标：x 向右、y 向下；正旋转角在屏幕上表现为顺时针。
机械结构若需要毫米量，仍需另做相机标定；本目录只输出题目要求的像素量。

如果检测已经得到 1～4 个正确多边形，但提示“未找到满足边长和邻接关系”，说明
输入碎片的边长无法按容差组成原项目规定的闭合邻接图。此时不应盲目增大容差，
应先检查碎片是否确实来自同一矩形，以及相机透视是否已足够小。

