import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 配置区域
# ==========================================

# 定义需要生成的波段组合
# 格式: "后缀名": (波段R, 波段G, 波段B)
# WorldView-2 常用组合:
BAND_CONFIGS = {
    "FalseColor": (7, 5, 3),  # 标准假彩色 (植被红，水体黑)
    "TrueColor": (5, 3, 2)  # 真彩色 (符合人眼视觉)
}

# 拉伸参数
STRETCH_RANGE = (2, 98)  # 2% - 98% 线性拉伸


# ==========================================
# 核心处理函数
# ==========================================

def generate_visualizations(tif_path_list):
    """
    批量处理输入的 TIFF 路径列表，生成可视化图像并保存在原目录下。
    """
    for tif_path in tif_path_list:
        if not os.path.exists(tif_path):
            print(f"❌ [跳过] 文件不存在: {tif_path}")
            continue

        print(f"\n📂 正在处理: {os.path.basename(tif_path)} ...")

        try:
            # 1. 读取影像 (只读取一次，提高效率)
            img = tifffile.imread(tif_path).astype(np.float32)

            # 维度调整 (H, W, C)
            if img.ndim == 3 and img.shape[0] < img.shape[1]:
                img = np.transpose(img, (1, 2, 0))

            # 获取原文件信息，用于生成保存路径
            dir_name = os.path.dirname(tif_path)
            file_name = os.path.splitext(os.path.basename(tif_path))[0]
            max_band = img.shape[2]

            # 2. 遍历配置，生成不同组合的图像
            for config_name, bands in BAND_CONFIGS.items():
                # 检查波段是否越界
                if any(b > max_band for b in bands):
                    print(f"   ⚠️ [跳过] {config_name}: 需要波段 {bands}，但图像只有 {max_band} 个波段")
                    continue

                # 提取波段 (转换为 0-based 索引)
                b_indices = [b - 1 for b in bands]
                composition = img[:, :, b_indices]

                # 图像增强 (百分比拉伸)
                vis_img = composition.copy()
                for i in range(3):
                    band_data = vis_img[:, :, i]
                    p_min, p_max = np.nanpercentile(band_data, STRETCH_RANGE)
                    if p_max == p_min:
                        vis_img[:, :, i] = 0
                    else:
                        vis_img[:, :, i] = (band_data - p_min) / (p_max - p_min)

                vis_img = np.clip(vis_img, 0, 1)

                # 3. 自动构建保存路径
                # 格式: 原文件名_类型_波段号.png
                # 例如: M0-1_mul_FalseColor_753.png
                band_str = "".join(map(str, bands))
                save_name = f"{file_name}_{config_name}_{band_str}.png"
                save_path = os.path.join(dir_name, save_name)

                # 保存
                plt.imsave(save_path, vis_img)
                print(f"   ✅ 已保存: {save_name}")

        except Exception as e:
            print(f"❌ 处理文件 {tif_path} 时发生错误: {e}")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    # 📝 在这里填写你需要处理的文件路径 (支持多个)
    # 只要是完整路径，脚本会自动把结果存在该路径对应的文件夹里

    input_files = [
        # 融合结果
        r'F:\2025UCAS\多源遥感图像融合\fusion\预测结果_Attention\Fused_Attn_M0-1_pan.tif',
        r'F:\2025UCAS\多源遥感图像融合\fusion\预测结果_Attention\Fused_Attn_M11-3_pan.tif',

        # 原始多光谱图 (如果不想要处理这些，注释掉即可)
        # r'F:\2025UCAS\多源遥感图像融合\fusion\真实测试图片\MS_up_800\M0-1_mul.tif',
        # r'F:\2025UCAS\多源遥感图像融合\fusion\真实测试图片\MS_up_800\M11-3_mul.tif',
    ]

    # 开始运行
    generate_visualizations(input_files)

    print("\n🎉 所有任务处理完成！")