import csv
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
import math


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    # 遥感影像通常是 11-bit (2047) 或 12-bit (4095)
    # 为了数值稳定性，建议固定一个归一化上限，而不是每张图动态计算
    MAX_VAL = 2047.0

    BATCH_SIZE = 16
    EPOCHS = 100
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_BANDS = 8  # 明确指定8通道

    # --- 路径配置 (请根据实际情况调整) ---
    ROOT_DIR = r'F:\2025UCAS\多源遥感图像融合\fusion'

    TRAIN_DIR = os.path.join(ROOT_DIR, r'训练数据集\train_data\train')
    VAL_DIR = os.path.join(ROOT_DIR, r'训练数据集\val_data\val')

    # 真实测试路径
    TEST_PAN_DIR = os.path.join(ROOT_DIR, r'真实测试图片\PAN_cut_800')
    TEST_MS_DIR = os.path.join(ROOT_DIR, r'真实测试图片\MS_up_800')

    RESULT_DIR = os.path.join(ROOT_DIR, r'预测结果')
    PLOT_DIR = os.path.join(ROOT_DIR, r'训练图表')  # 新增图表保存路径

    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)


# ==========================================
# 2. 评价指标计算工具 (Metrics)
# ==========================================
class Metrics:
    @staticmethod
    def calculate_psnr(img1, img2, data_range=1.0):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 10 * math.log10((data_range ** 2) / mse)

    @staticmethod
    def calculate_ssim(img1, img2, data_range=1.0):
        # img shape: (H, W, C)
        total_ssim = 0
        channels = img1.shape[2]
        for i in range(channels):
            total_ssim += ssim_func(img1[:, :, i], img2[:, :, i], data_range=data_range)
        return total_ssim / channels

    @staticmethod
    def calculate_sam(img1, img2):
        # Spectral Angle Mapper (SAM)
        # img shape: (H, W, C) -> transform to (N, C)
        H, W, C = img1.shape
        v1 = img1.reshape(-1, C)
        v2 = img2.reshape(-1, C)

        # Dot product
        dot = np.sum(v1 * v2, axis=1)
        norm1 = np.sqrt(np.sum(v1 ** 2, axis=1))
        norm2 = np.sqrt(np.sum(v2 ** 2, axis=1))

        # Clip to avoid nan/inf errors
        cos_theta = dot / (norm1 * norm2 + 1e-8)
        cos_theta = np.clip(cos_theta, -1, 1)

        sam = np.arccos(cos_theta)
        # Convert to degrees and mean
        return np.mean(sam) * (180 / math.pi)

    @staticmethod
    def calculate_ergas(img_fake, img_real, scale_ratio=4):
        # Erreur Relative Globale Adimensionnelle de Synthèse
        # input shape: (H, W, C)
        H, W, C = img_fake.shape
        mse = np.mean((img_fake - img_real) ** 2, axis=(0, 1))
        mean_real = np.mean(img_real, axis=(0, 1))

        sum_ergas = np.sum((mse) / (mean_real ** 2 + 1e-8))
        return 100 / scale_ratio * np.sqrt((1 / C) * sum_ergas)


# ==========================================
# 3. 数据集加载器
# ==========================================
class FusionDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.mode = mode
        # 严格匹配文件名，防止乱序
        self.pan_files = sorted(glob.glob(os.path.join(root_dir, '*_pan.tif')))
        self.mul_files = sorted(glob.glob(os.path.join(root_dir, '*_mul.tif')))
        self.gt_files = sorted(glob.glob(os.path.join(root_dir, '*_ymul.tif')))

        # 简单校验
        # 如果是验证模式，允许数量为0(可能是还没切分)；如果是训练模式必须有数据
        if mode == 'train':
            assert len(self.pan_files) > 0, f"错误：{root_dir} 下没有找到训练数据！"
        if len(self.pan_files) != len(self.mul_files) or len(self.pan_files) != len(self.gt_files):
            print(
                f"警告：{root_dir} 下的文件数量不匹配！PAN:{len(self.pan_files)}, MUL:{len(self.mul_files)}, GT:{len(self.gt_files)}")

    def __len__(self):
        return len(self.pan_files)

    def __getitem__(self, idx):
        pan = tifffile.imread(self.pan_files[idx]).astype(np.float32)
        mul = tifffile.imread(self.mul_files[idx]).astype(np.float32)
        gt = tifffile.imread(self.gt_files[idx]).astype(np.float32)

        # 归一化 (统一除以 MAX_VAL，保持物理意义一致性)
        pan = pan / Config.MAX_VAL
        mul = mul / Config.MAX_VAL
        gt = gt / Config.MAX_VAL

        # 维度调整 (H, W) -> (1, H, W)
        if pan.ndim == 2:
            pan = pan[np.newaxis, :, :]

        # 维度调整 (H, W, C) -> (C, H, W)
        if mul.ndim == 3 and mul.shape[2] < mul.shape[0]:  # HWC to CHW
            mul = np.transpose(mul, (2, 0, 1))
        if gt.ndim == 3 and gt.shape[2] < gt.shape[0]:
            gt = np.transpose(gt, (2, 0, 1))

        return torch.from_numpy(pan), torch.from_numpy(mul), torch.from_numpy(gt)


# 真实测试集加载器 (保持类似 v2，但增加了配对检查)
class RealTestDataset(Dataset):
    def __init__(self, pan_dir, ms_dir):
        self.pan_files = sorted(glob.glob(os.path.join(pan_dir, '*.tif')))
        self.ms_files = sorted(glob.glob(os.path.join(ms_dir, '*.tif')))
        # 可以在这里增加文件名匹配逻辑，这里暂且假设排序后对应

    def __len__(self):
        return min(len(self.pan_files), len(self.ms_files))

    def __getitem__(self, idx):
        p_path = self.pan_files[idx]
        m_path = self.ms_files[idx]

        pan = tifffile.imread(p_path).astype(np.float32)
        mul = tifffile.imread(m_path).astype(np.float32)

        # 记录每张图自己的最大值用于还原，因为是展示用
        local_max = Config.MAX_VAL

        pan = pan / local_max
        mul = mul / local_max

        if pan.ndim == 2: pan = pan[np.newaxis, :, :]
        # HWC -> CHW
        if mul.ndim == 3 and mul.shape[2] <= 10:
            mul = np.transpose(mul, (2, 0, 1))

        pan_t = torch.from_numpy(pan)
        mul_t = torch.from_numpy(mul)

        # 上采样适配
        if mul_t.shape[-1] != pan_t.shape[-1]:
            mul_t = torch.nn.functional.interpolate(mul_t.unsqueeze(0), size=pan_t.shape[1:], mode='bicubic').squeeze(0)

        return pan_t, mul_t, os.path.basename(p_path), local_max


# ==========================================
# 4. 模型定义 (PanNet)
# ==========================================
class PanNet(nn.Module):
    def __init__(self, spectral_bands=8):
        super(PanNet, self).__init__()
        # Pan(1) + MS(8) = 9 channels input
        self.conv1 = nn.Conv2d(spectral_bands + 1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.res2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(32, spectral_bands, kernel_size=3, padding=1)

    def forward(self, pan, mul):
        # 简单的残差学习
        x = torch.cat([pan, mul], dim=1)
        feat = self.relu(self.conv1(x))
        feat = self.relu(self.res1(feat))
        feat = self.relu(self.res2(feat))
        residual = self.conv_out(feat)
        return mul + residual


# ==========================================
# 5. 训练与验证主流程
# ==========================================
def train_and_validate():
    print(f"载入训练集: {Config.TRAIN_DIR}")
    train_dataset = FusionDataset(Config.TRAIN_DIR, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    print(f"载入验证集: {Config.VAL_DIR}")
    # 确保验证集有数据
    val_dataset = FusionDataset(Config.VAL_DIR, mode='val')
    if len(val_dataset) == 0:
        print("⚠️ 警告：验证集为空，将跳过验证步骤。")
        val_loader = None
    else:
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # 验证时Batch=1方便算指标

    model = PanNet(spectral_bands=Config.NUM_BANDS).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.L1Loss()

    # 记录历史数据用于绘图
    history = {
        'train_loss': [], 'val_loss': [],
        'psnr': [], 'ssim': [], 'sam': [], 'ergas': []
    }

    print("=== 开始训练 ===")
    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        epoch_loss = 0
        for pan, mul, gt in train_loader:
            pan, mul, gt = pan.to(Config.DEVICE), mul.to(Config.DEVICE), gt.to(Config.DEVICE)

            optimizer.zero_grad()
            output = model(pan, mul)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation ---
        avg_val_loss = 0
        avg_psnr = 0
        if val_loader:
            model.eval()
            val_loss = 0
            metric_psnr = 0
            metric_ssim = 0
            metric_sam = 0
            metric_ergas = 0

            with torch.no_grad():
                for pan, mul, gt in val_loader:
                    pan, mul, gt = pan.to(Config.DEVICE), mul.to(Config.DEVICE), gt.to(Config.DEVICE)
                    output = model(pan, mul)

                    # Calculate Loss
                    loss = criterion(output, gt)
                    val_loss += loss.item()

                    # 准备计算指标 (转回 CPU numpy, CHW -> HWC)
                    out_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    gt_np = gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                    # Clip values to [0, 1] for metric calculation
                    out_np = np.clip(out_np, 0, 1)
                    gt_np = np.clip(gt_np, 0, 1)

                    metric_psnr += Metrics.calculate_psnr(out_np, gt_np)
                    metric_ssim += Metrics.calculate_ssim(out_np, gt_np)
                    metric_sam += Metrics.calculate_sam(out_np, gt_np)
                    metric_ergas += Metrics.calculate_ergas(out_np, gt_np)

            avg_val_loss = val_loss / len(val_loader)
            avg_psnr = metric_psnr / len(val_loader)
            avg_ssim = metric_ssim / len(val_loader)
            avg_sam = metric_sam / len(val_loader)
            avg_ergas = metric_ergas / len(val_loader)

        history['val_loss'].append(avg_val_loss)
        history['psnr'].append(avg_psnr)
        history['ssim'].append(avg_ssim)
        history['sam'].append(avg_sam)
        history['ergas'].append(avg_ergas)

        log_msg = f"Epoch [{epoch + 1}/{Config.EPOCHS}] Train Loss: {avg_train_loss:.5f}"
        if val_loader:
            log_msg += f" | Val Loss: {avg_val_loss:.5f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}"
        print(log_msg)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(Config.RESULT_DIR, 'pannet_final.pth'))
    print("训练结束，模型已保存。")

    # 绘图
    plot_metrics(history)


# ==========================================
# 6. 绘图函数
# ==========================================
def plot_metrics(history):
    epochs = range(1, Config.EPOCHS + 1)

    # 1. Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    if any(history['val_loss']):
        plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.PLOT_DIR, 'loss_curve.png'))
    plt.close()

    # 2. 综合指标曲线 (2x2 subplot) - 如果没有验证集则跳过
    if not any(history['psnr']): return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # PSNR (越高越好)
    axs[0, 0].plot(epochs, history['psnr'], 'g-')
    axs[0, 0].set_title('PSNR')
    # axs[0, 0].set_xlabel('Epochs')
    # axs[0, 0].grid(True)

    # SSIM (越高越好)
    axs[0, 1].plot(epochs, history['ssim'], 'b-')
    axs[0, 1].set_title('SSIM')
    # axs[0, 1].set_xlabel('Epochs')
    # axs[0, 1].grid(True)

    # SAM (越低越好)
    axs[1, 0].plot(epochs, history['sam'], 'r-')
    axs[1, 0].set_title('SAM')
    # axs[1, 0].set_xlabel('Epochs')
    # axs[1, 0].grid(True)

    # ERGAS (越低越好)
    axs[1, 1].plot(epochs, history['ergas'], 'm-')
    axs[1, 1].set_title('ERGAS')
    # axs[1, 1].set_xlabel('Epochs')
    # axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOT_DIR, 'metrics_pannet.png'))
    plt.close()
    print(f"图表已保存至: {Config.PLOT_DIR}")

    # --- 2. 输出第100轮（最后一轮）指标 ---
    # 获取列表中的最后一个元素 [-1] 即为最终结果
    last_epoch = len(history['train_loss'])
    final_psnr = history['psnr'][-1]
    final_ssim = history['ssim'][-1]
    final_sam = history['sam'][-1]
    final_ergas = history['ergas'][-1]
    final_t_loss = history['train_loss'][-1]
    final_v_loss = history['val_loss'][-1]

    print("\n" + "=" * 40)
    print(f"最终训练结果 (第 {last_epoch} 轮)")
    print("=" * 40)
    print(f"Train Loss : {final_t_loss:.6f}")
    print(f"Val Loss   : {final_v_loss:.6f}")
    print("-" * 40)
    print(f"PSNR       : {final_psnr:.4f}")
    print(f"SSIM       : {final_ssim:.4f}")
    print(f"SAM        : {final_sam:.4f}")
    print(f"ERGAS      : {final_ergas:.4f}")
    print("=" * 40 + "\n")

    # --- 3. 保存为 CSV 表格方便复制 ---
    csv_path = os.path.join(Config.PLOT_DIR, 'final_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['Epoch', 'PSNR', 'SSIM', 'SAM', 'ERGAS', 'Train_Loss', 'Val_Loss'])
        # 写入数据
        writer.writerow([last_epoch, f"{final_psnr:.4f}", f"{final_ssim:.4f}",
                         f"{final_sam:.4f}", f"{final_ergas:.4f}",
                         f"{final_t_loss:.6f}", f"{final_v_loss:.6f}"])

    print(f"指标数据已保存至: {csv_path}")



# ==========================================
# 7. 真实大图推理
# ==========================================
def inference():
    print("正在处理真实测试集大图...")
    dataset = RealTestDataset(Config.TEST_PAN_DIR, Config.TEST_MS_DIR)
    if len(dataset) == 0:
        print("⚠️ 真实测试集文件夹为空，跳过推理。")
        return

    model = PanNet(spectral_bands=Config.NUM_BANDS).to(Config.DEVICE)
    model_path = os.path.join(Config.RESULT_DIR, 'pannet_final.pth')
    if not os.path.exists(model_path):
        print(f"⚠️ 未找到模型文件 {model_path}，请先进行训练。")
        return

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            pan, mul, fname, max_val = dataset[i]
            pan = pan.unsqueeze(0).to(Config.DEVICE)
            mul = mul.unsqueeze(0).to(Config.DEVICE)

            # 推理
            fused = model(pan, mul)

            # 后处理
            fused = fused.squeeze(0).cpu().numpy()  # CHW
            fused = fused * max_val
            fused = np.clip(fused, 0, 65535).astype(np.uint16)
            fused = np.transpose(fused, (1, 2, 0))  # HWC

            save_path = os.path.join(Config.RESULT_DIR, f"Fused_{fname}")
            tifffile.imwrite(save_path, fused)
            print(f"[{i + 1}/{len(dataset)}] 保存: {save_path}")


# ==========================================
# 8. 生成误差热力图
# ==========================================
def generate_error_map_assets(model_path, save_dir, sample_index=0):
    """
    使用验证集数据，计算预测结果与Ground Truth之间的绝对误差，并绘制热力图。

    参数:
        model_path: 训练好的模型权重路径 (.pth)
        save_dir: 热力图保存文件夹
        sample_index: 选择验证集中的第几张图片进行分析 (默认第0张)
    """
    print(f"\n=== 正在生成误差热力图 (验证集样本 Index: {sample_index}) ===")

    # 1. 准备数据 (使用验证集，因为需要 Ground Truth)
    # 这里 mode='val' 虽然在v3的FusionDataset里没特殊逻辑，但保持语义清晰
    dataset = FusionDataset(Config.VAL_DIR, mode='val')
    if len(dataset) <= sample_index:
        print(f"❌ 错误: 验证集中只有 {len(dataset)} 张图片，无法读取索引 {sample_index}")
        return

    # 读取数据 (已归一化到 0-1)
    pan, mul, gt = dataset[sample_index]

    # 2. 加载模型与推理
    if not os.path.exists(model_path):
        print(f"❌ 错误：未找到模型文件 {model_path}，请确认路径或先进行训练。")
        return

    model = PanNet(spectral_bands=Config.NUM_BANDS).to(Config.DEVICE)
    # map_location 确保在只有CPU的机器上也能加载GPU训练的模型
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    with torch.no_grad():
        # 增加 batch 维度并移至设备
        pan_t = pan.unsqueeze(0).to(Config.DEVICE)
        mul_t = mul.unsqueeze(0).to(Config.DEVICE)

        # 预测
        fused = model(pan_t, mul_t)

    # 3. 数据转换 (Tensor -> Numpy, CHW -> HWC)
    # 关键：反归一化 (* Config.MAX_VAL)，恢复到真实的像素值量级，这样计算的误差才有物理意义
    pred_np = fused.squeeze(0).cpu().numpy().transpose(1, 2, 0) * Config.MAX_VAL
    gt_np = gt.cpu().numpy().transpose(1, 2, 0) * Config.MAX_VAL

    img_h, img_w, img_c = pred_np.shape
    print(f"当前处理图像尺寸: {img_w}x{img_h}, 波段数: {img_c}")

    # 4. 计算残差热力图 (核心步骤)
    # Error = |Predicted - GT| (绝对误差)
    diff = np.abs(pred_np - gt_np)
    # 将所有波段的误差取平均，得到一张单通道的“综合误差图”
    error_map = np.mean(diff, axis=2)

    # 设置热力图显示的上限 (vmax)。
    # 对于11-bit数据(max 2047)，如果平均误差超过200 (约10%) 就算比较大了。
    # 设置一个合理的上限可以让红色的区域更突出。
    heatmap_vmax = 0.1 * Config.MAX_VAL

    # 5. 绘图并保存
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    # 使用 'jet' 颜色映射：蓝=低误差，红=高误差
    im = plt.imshow(error_map, cmap='jet', vmin=0, vmax=heatmap_vmax)
    plt.colorbar(im, label=f'Mean Absolute Error (Pixel Value, 0-{int(heatmap_vmax)})')
    plt.title(f'Error Heatmap (Validation Sample {sample_index})')
    plt.axis('off')

    save_name = f'ErrorMap_PanNet_Sample{sample_index}.png'
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ 已保存误差热力图至: {save_path}")


# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    # --- 阶段 1: 训练 (如果已经训练过，可注释) ---
    train_and_validate()

    # --- 阶段 2: 真实数据推理 ---
    # inference()

    # # --- 阶段 3: [新增] 生成误差热力图 ---
    # # 确保你已经训练过模型，并且该路径下有 pannet_final.pth 文件
    # model_weights = os.path.join(Config.RESULT_DIR, 'pannet_final.pth')
    # # 图片保存位置
    # heatmap_dir = os.path.join(Config.ROOT_DIR, '汇报素材_pannet')
    # if not os.path.exists(heatmap_dir): os.makedirs(heatmap_dir)
    # # 运行热力图生成
    # # 你可以修改 sample_index 来查看验证集中不同图片的效果
    # # 建议多跑几张 (例如 index=0, 1, 2) 挑典型的
    # # for i in range(10, 20):
    # #     generate_error_map_assets(model_weights, heatmap_dir, sample_index=i)


    # generate_error_map_assets(model_weights, heatmap_dir, sample_index=1) # 可以取消注释跑更多
