#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
航空遥感DOM tile 全局统一拉伸脚本（消除切块后色差）
- 先从原始大DOM计算全局 2%~98% 统计范围
- 对所有已生成的tile应用完全相同的拉伸参数，转为 uint8
- 输出带 _stretched 后缀的新tile，保留地理信息和LZW压缩
"""

import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from tqdm import tqdm


# ====================== 需要修改的参数 ======================
original_dom_path = r"/mnt/f/forest_agent_project/input/GX_dong_men_lin_chang/anshu_plantation.tif"   # ← 改成你的原始DOM完整路径

tiles_input_dir   = r"/mnt/f/forest_agent_project/outputs/GX_dong_men_lin_chang/anshu_plantation_tiles_100m/tiles"   # ← 注意前面加 /mnt/f/

stretched_output_dir = r"/mnt/f/forest_agent_project/outputs/GX_dong_men_lin_chang/anshu_plantation_tiles_100m/stretched_tiles"  # 输出文件夹建议也放在同一位置
# ============================================================
percent_low  = 2.0     # 低百分位（推荐2）
percent_high = 98.0    # 高百分位（推荐98）
# ============================================================

def calculate_global_scale_params(dom_path: str, p_low=2.0, p_high=98.0):
    """从原始DOM计算每个波段的全局拉伸参数"""
    print(f"正在从原始影像计算全局统计: {dom_path}")
    with rasterio.open(dom_path) as src:
        scale_params = []
        for b in range(1, src.count + 1):
            # 读取波段数据（masked=True 自动处理 nodata）
            data = src.read(b, masked=True)
            valid_data = data.compressed()  # 只取有效值
            
            if len(valid_data) == 0:
                low = 0.0
                high = 1.0
            else:
                low = np.percentile(valid_data, p_low)
                high = np.percentile(valid_data, p_high)
            
            # 避免 low == high 的极端情况
            if abs(high - low) < 1e-6:
                high = low + 1.0
            
            scale_params.append((float(low), float(high)))
            print(f"  Band {b}: {low:.2f} ~ {high:.2f} (percentile {p_low}% ~ {p_high}%)")
        
        print(f"全局拉伸参数计算完成，共 {src.count} 个波段\n")
        return scale_params


def stretch_tile(tile_path: Path, output_path: Path, scale_params: list):
    """对单个tile应用全局拉伸"""
    with rasterio.open(tile_path) as src:
        data = src.read()                    # shape: (C, H, W)
        
        stretched = np.zeros_like(data, dtype=np.uint8)
        
        for i in range(src.count):
            low, high = scale_params[i]
            band = data[i].astype(np.float32)
            # 线性拉伸 + clip 到 0-255
            scaled = (band - low) / (high - low) * 255.0
            stretched[i] = np.clip(scaled, 0, 255).astype(np.uint8)
        
        # 更新profile
        profile = src.profile.copy()
        profile.update(
            dtype='uint8',
            compress='LZW',
            photometric='RGB' if src.count == 3 else None,
            nodata=0
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(stretched)
            # 明确设置RGB颜色解释（提升兼容性）
            if src.count == 3:
                dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
    
    return output_path.name


def main():
    input_dir = Path(tiles_input_dir)
    output_dir = Path(stretched_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"错误：输入文件夹不存在: {input_dir}")
        sys.exit(1)

    # 1. 计算全局拉伸参数
    scale_params = calculate_global_scale_params(original_dom_path, percent_low, percent_high)

    # 2. 获取所有 .tif 文件
    tile_files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.TIF")))
    if not tile_files:
        print("错误：在输入文件夹中未找到 .tif 文件")
        sys.exit(1)

    print(f"找到 {len(tile_files)} 个 tile，开始批量拉伸...\n")

    # 3. 批量处理
    for tile_path in tqdm(tile_files, desc="拉伸进度", unit="tile"):
        out_name = tile_path.stem + "_stretched.tif"
        output_path = output_dir / out_name
        
        try:
            stretch_tile(tile_path, output_path, scale_params)
        except Exception as e:
            print(f"\n处理失败 {tile_path.name}: {e}")

    print("\n处理完成！")
    print(f"输出路径: {output_dir}")
    print("建议：在QGIS中加载几个相邻的_stretched.tif文件，与原始DOM对比查看色差是否消除。")


if __name__ == "__main__":
    main()