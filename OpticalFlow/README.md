# Optical Flow-Based 3D Image Signal Enhancement for Dendrites and Spines

基于光流算法的3D显微镜图像信号增强，用于增强树突和树突棘信号。

## 项目概述

### 光流方法
采用 **Dual TV-L1 光流算法**（Total Variation L1），这是一种高精度变分方法：
- 提供精确的稠密光流
- 保持运动边界
- 对噪声和光照变化鲁棒
- 适合精细生物结构分析

## 功能说明

### 信号增强 (Signal Enhancement)
**主要目标**：增强树突和树突棘信号  
**次要目标**：压暗背景

**原理**：
1. 计算相邻Z层之间的光流
2. 分析光流一致性
3. 一致的光流 → 真实结构（树突/树突棘穿过Z轴）
4. 不一致的光流 → 背景
5. **强力增强**信号区域，可选地抑制背景

**特点**：
- 非线性自适应增强（信号越强，增强越多）
- 局部对比度增强（CLAHE）
- 边缘保持平滑
- 可调节的增强强度（推荐2-5倍）

## 项目结构

```
OpticalFlow/
├── signal_enhancement.py      # 信号增强主程序
├── utils/                      # 工具函数
│   ├── io_utils.py            # I/O操作
│   └── opticalflow_utils.py   # 光流计算
├── outputs/                    # 输出目录
├── requirements.txt           # 依赖
└── README.md                  # 本文件
```

## 安装

```bash
# 创建conda环境
conda create -n opticalflow python=3.9
conda activate opticalflow

# 安装依赖
pip install -r requirements.txt
```

**依赖**：
- Python 3.8+
- OpenCV with contrib (包含optflow模块)
- NumPy, SciPy, scikit-image
- Tifffile (处理3D TIF)
- Matplotlib (可视化)

## 使用方法

### 信号增强

**标准命令（单行）**：
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --enhance 3.0 --power 1.5 --bg-suppress 0.7
```

**使用Farneback方法（更快）**：
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --method farneback --enhance 3.0 --power 1.5 --bg-suppress 0.7
```

**参数说明**：
- `--input`: 输入3D TIF文件
- `--output`: 输出增强后的TIF文件
- `--method`: 光流方法 (`tvl1` 或 `farneback`，默认tvl1)
- `--threshold`: 信号阈值 (0-1, 越低越敏感, 默认0.3)
- `--percentile`: 使用百分位阈值替代绝对阈值 (例如90表示top 10%)
- `--window`: 一致性窗口大小 (默认5)
- `--enhance`: **增强强度** (推荐2-5, 默认3.0)
- `--power`: 非线性增强幂次 (>1, 默认1.5)
- `--bg-suppress`: 背景抑制 (0-1, 1=不抑制, 默认0.7)
- `--no-adaptive`: 禁用自适应增强
- `--edge-sigma`: 边缘保持平滑参数 (默认0.1)
- `--no-roi-clahe`: 对整个slice应用CLAHE而非仅ROI
- `--no-vis`: 禁用可视化

**输出文件**：
- 增强后的体积 (TIF)
- 信号强度图 (TIF)
- 对比可视化 (PNG)
- 信号掩码 (PNG)
- 增强因子图 (PNG)

**推荐参数组合**：

1. **保守增强**（适合已有较好信号的图像）：
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --enhance 2.0 --power 1.2 --bg-suppress 0.8
```

2. **标准增强**（默认，适合大多数情况）：
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --enhance 3.0 --power 1.5 --bg-suppress 0.7
```

3. **强力增强**（适合信号较弱的图像）：
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --enhance 5.0 --power 2.0 --bg-suppress 0.5
```

4. **使用百分位阈值（自适应）**：
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --percentile 90 --enhance 3.0
```

## 算法细节

### 信号增强策略

1. **光流一致性分析**：
```python
signal_strength = mean_magnitude / (std_magnitude + mean_magnitude * 0.1 + ε)
```
- 高一致性 → 稳定光流 → 真实结构
- 低一致性 → 随机光流 → 背景

2. **自适应增强**：
```python
enhancement_map = 1.0 + (strength - 1.0) * signal_strength^power
```
- 信号强度越高，增强越多
- 非线性增强保护弱信号

3. **对比度增强**：
- 在信号区域应用CLAHE
- 增强局部对比度
- 突出树突棘等细节结构

4. **边缘保持平滑**：
- 双边滤波器减少伪影
- 保持结构边界清晰

### 性能考虑

**计算成本**：
- TV-L1: 高精度，较慢 (~5-10秒/层对)
- Farneback: 较快，略低精度 (~1-2秒/层对)

**内存需求**：
- 典型120MB TIF (512×512×100): ~2GB峰值内存
- 光流体积: ~1GB
- 可按层处理更大体积

**优化建议**：
1. 使用GPU加速 (OpenCV CUDA)
2. 减少TV-L1迭代次数
3. 处理ROI而非完整体积
4. 原型用Farneback，最终结果用TV-L1

## 示例

### 示例1：标准信号增强
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --enhance 3.0 --power 1.5
```

### 示例2：强力增强（弱信号）
```powershell
python signal_enhancement.py --input ../Data/test/F13_2_20250325_roi1_Red.tif --output outputs/F13_2_enhanced.tif --enhance 5.0 --power 2.0 --bg-suppress 0.5
```

### 示例3：使用Farneback快速处理
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --method farneback --enhance 3.0
```

## 故障排除

### 问题：找不到 'cv2.optflow' 模块
**解决**：安装 `opencv-contrib-python`
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### 问题：内存不足
**解决**：处理较小的ROI
```python
# 处理中心ROI
volume_roi = volume[:, 100:400, 100:400]
```

### 问题：处理速度慢
**解决**：使用Farneback方法（快5-10倍）
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --method farneback --enhance 3.0
```

### 问题：增强效果不明显
**解决**：增加增强强度和非线性幂次
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --enhance 5.0 --power 2.0 --bg-suppress 0.5
```

### 问题：背景过暗
**解决**：增加背景抑制参数
```powershell
python signal_enhancement.py --input ../Data/test/F08_1_20240817_roi3_Red.tif --output outputs/F08_1_enhanced.tif --bg-suppress 0.9 --enhance 3.0
```

## 参考文献

1. **Dual TV-L1 Optical Flow**:
   - Zach, C., Pock, T., & Bischof, H. (2007). "A Duality Based Approach for Realtime TV-L1 Optical Flow"

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
   - Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization"

3. **Biological Applications**:
   - Optical flow for neurite tracking in microscopy
   - Motion analysis in live-cell imaging

## 许可

本项目是约翰霍普金斯大学SpineDetect研究项目的一部分。

## 联系

如有问题，请联系开发团队。
