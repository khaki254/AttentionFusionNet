import csv
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    MAX_VAL = 2047.0
    BATCH_SIZE = 16
    EPOCHS = 200
    LR = 5e-4  # 初始学习率稍微降低
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_BANDS = 8

    # 路径 (保持不变)
    ROOT_DIR = r'F:\2025UCAS\多源遥感图像融合\fusion'
    TRAIN_DIR = os.path.join(ROOT_DIR, r'训练数据集\train_data\train')
    VAL_DIR = os.path.join(ROOT_DIR, r'训练数据集\val_data\val')
    TEST_PAN_DIR = os.path.join(ROOT_DIR, r'真实测试图片\PAN_cut_800')
    TEST_MS_DIR = os.path.join(ROOT_DIR, r'真实测试图片\MS_up_800')
    RESULT_DIR = os.path.join(ROOT_DIR, r'预测结果_Attention')  # 区分结果文件夹
    PLOT_DIR = os.path.join(ROOT_DIR, r'训练图表_Attention')

    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)


# ==========================================
# 2. 创新模块: 注意力机制 (CBAM Block)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        # 1. 压缩：把二维特征图变成一个点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 每个通道只输出 1 个值（对应 1×1 的特征图）
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 2. 激励：多层感知机 (MLP)
        # 利用 1x1 卷积代替全连接层，为了减少参数，先降维再升维
        # in_planes (64) -> in_planes // ratio (8)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        # in_planes // ratio (8) -> in_planes (64)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # 3. 归一化：把输出变成 0~1 之间的权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # padding 的设置是为了保证卷积后，图像的长宽不变 (保持 Same Padding)
        # 如果 kernel_size=7, padding=3 -> (W - 7 + 2*3)/1 + 1 = W
        padding = 3 if kernel_size == 7 else 1
        # 定义一个 2通道 -> 1通道 的卷积层
        # 把 AvgPool 和 MaxPool 的结果拼起来
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # 归一化激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # --- 步骤 1: 沿通道维度的压缩 ---
        # 我们不关心这 64 个通道具体是什么，我们只想知道在 (h, w) 这个像素点上，
        # 整体的响应强度是多少。

        # 平均池化: 衡量该像素点的“平均信息量”
        # dim=1 表示压缩通道维度 (Channel dim)
        # [B, 64, H, W] -> [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # 最大池化: 衡量该像素点的“最显著特征”
        # (比如这个点在某个通道上响应极其强烈，那就是纹理点)
        # [B, 64, H, W] -> [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # --- 步骤 2: 拼接 ---
        # 将“平均特征”和“最大特征”叠在一起
        # [B, 1, H, W] + [B, 1, H, W] -> [B, 2, H, W]
        x = torch.cat([avg_out, max_out], dim=1)

        # --- 步骤 3: 卷积融合 ---
        # 利用 7x7 卷积，根据这两个特征图，学习出哪里重要
        # [B, 2, H, W] -> [B, 1, H, W]
        x = self.conv1(x)

        # --- 步骤 4: 生成掩膜 ---
        # 将数值映射到 0~1 之间，形成最终的“空间注意力图” (Spatial Attention Map)
        return self.sigmoid(x)


# ==========================================
# 3. 创新模型: AttentionFusionNet
# ==========================================
class AttentionFusionNet(nn.Module):
    def __init__(self, spectral_bands=8):
        super(AttentionFusionNet, self).__init__()

        # 特征提取层
        self.conv_in = nn.Conv2d(spectral_bands + 1, 64, kernel_size=3, padding=1)

        # 深度残差块 + 注意力
        self.res_block1 = self._make_layer(64)
        self.res_block2 = self._make_layer(64)
        self.res_block3 = self._make_layer(64)

        # 空间注意力 (关注纹理细节)
        self.sa = SpatialAttention()
        # 通道注意力 (关注光谱保真)
        self.ca = ChannelAttention(64)

        # 重建层
        self.conv_out = nn.Conv2d(64, spectral_bands, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, dim):
        return nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, pan, mul):
        # 1. 输入拼接
        x = torch.cat([pan, mul], dim=1)

        # 2. 浅层特征提取
        feat = self.relu(self.conv_in(x))

        # 3. 深层特征提取 (残差学习)
        res1 = self.res_block1(feat)
        feat = feat + res1

        res2 = self.res_block2(feat)
        feat = feat + res2

        # 4. 关键创新: 混合注意力修正
        # 空间注意力：让网络关注边缘
        feat = feat * self.sa(feat)
        # 通道注意力：让网络校准光谱
        feat = feat * self.ca(feat)

        # 5. 残差重建
        residual = self.conv_out(feat)

        # 6. 输出 (跳跃连接)
        return mul + residual


# ==========================================
# 4. 创新损失: 混合损失 (L1 + Gradient)
# ==========================================
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, pred, gt):
        # 对每个波段分别计算梯度
        b, c, h, w = pred.shape
        loss = 0
        for i in range(c):
            pred_ch = pred[:, i:i + 1, :, :]
            gt_ch = gt[:, i:i + 1, :, :]

            # 使用 Sobel 算子
            grad_x_pred = F.conv2d(pred_ch, self.weight_x.to(pred.device), padding=1)
            grad_y_pred = F.conv2d(pred_ch, self.weight_y.to(pred.device), padding=1)
            grad_pred = torch.abs(grad_x_pred) + torch.abs(grad_y_pred)

            grad_x_gt = F.conv2d(gt_ch, self.weight_x.to(gt.device), padding=1)
            grad_y_gt = F.conv2d(gt_ch, self.weight_y.to(gt.device), padding=1)
            grad_gt = torch.abs(grad_x_gt) + torch.abs(grad_y_gt)

            loss += F.l1_loss(grad_pred, grad_gt)
        return loss / c


class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()

    def forward(self, pred, gt):
        # 权重配比: 1.0 内容损失 + 0.1 边缘纹理损失
        return self.l1(pred, gt) + 0.1 * self.grad(pred, gt)


# ==========================================
# 5. 数据集工具
# ==========================================
class FusionDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.pan_files = sorted(glob.glob(os.path.join(root_dir, '*_pan.tif')))
        self.mul_files = sorted(glob.glob(os.path.join(root_dir, '*_mul.tif')))
        self.gt_files = sorted(glob.glob(os.path.join(root_dir, '*_ymul.tif')))

    def __len__(self):
        return len(self.pan_files)

    def __getitem__(self, idx):
        pan = tifffile.imread(self.pan_files[idx]).astype(np.float32) / Config.MAX_VAL
        mul = tifffile.imread(self.mul_files[idx]).astype(np.float32) / Config.MAX_VAL
        gt = tifffile.imread(self.gt_files[idx]).astype(np.float32) / Config.MAX_VAL
        if pan.ndim == 2: pan = pan[np.newaxis, :, :]
        if mul.ndim == 3 and mul.shape[2] < mul.shape[0]: mul = np.transpose(mul, (2, 0, 1))
        if gt.ndim == 3 and gt.shape[2] < gt.shape[0]: gt = np.transpose(gt, (2, 0, 1))
        return torch.from_numpy(pan), torch.from_numpy(mul), torch.from_numpy(gt)


class RealTestDataset(Dataset):
    def __init__(self, pan_dir, ms_dir):
        self.pan_files = sorted(glob.glob(os.path.join(pan_dir, '*.tif')))
        self.ms_files = sorted(glob.glob(os.path.join(ms_dir, '*.tif')))

    def __len__(self):
        return min(len(self.pan_files), len(self.ms_files))

    def __getitem__(self, idx):
        pan = tifffile.imread(self.pan_files[idx]).astype(np.float32)
        mul = tifffile.imread(self.ms_files[idx]).astype(np.float32)
        local_max = Config.MAX_VAL
        pan = pan / local_max
        mul = mul / local_max
        if pan.ndim == 2: pan = pan[np.newaxis, :, :]
        if mul.ndim == 3 and mul.shape[2] <= 10: mul = np.transpose(mul, (2, 0, 1))
        pan_t = torch.from_numpy(pan)
        mul_t = torch.from_numpy(mul)
        if mul_t.shape[-1] != pan_t.shape[-1]:
            mul_t = torch.nn.functional.interpolate(mul_t.unsqueeze(0), size=pan_t.shape[1:], mode='bicubic').squeeze(0)
        return pan_t, mul_t, os.path.basename(self.pan_files[idx]), local_max


class Metrics:
    @staticmethod
    def calculate_psnr(img1, img2, data_range=1.0):
        mse = np.mean((img1 - img2) ** 2)
        return 100 if mse == 0 else 10 * math.log10((data_range ** 2) / mse)

    @staticmethod
    def calculate_ssim(img1, img2, data_range=1.0):
        total_ssim = 0
        for i in range(img1.shape[2]):
            total_ssim += ssim_func(img1[:, :, i], img2[:, :, i], data_range=data_range)
        return total_ssim / img1.shape[2]

    @staticmethod
    def calculate_sam(img1, img2):
        H, W, C = img1.shape
        v1, v2 = img1.reshape(-1, C), img2.reshape(-1, C)
        dot = np.sum(v1 * v2, axis=1)
        norm = np.sqrt(np.sum(v1 ** 2, axis=1)) * np.sqrt(np.sum(v2 ** 2, axis=1))
        sam = np.arccos(np.clip(dot / (norm + 1e-8), -1, 1))
        return np.mean(sam) * (180 / math.pi)

    @staticmethod
    def calculate_ergas(img_fake, img_real, scale_ratio=4):
        mse = np.mean((img_fake - img_real) ** 2, axis=(0, 1))
        mean_real = np.mean(img_real, axis=(0, 1))
        return 100 / scale_ratio * np.sqrt(np.mean(mse / (mean_real ** 2 + 1e-8)))


# ==========================================
# 6. 训练与验证 (加入 Scheduler 和 HybridLoss)
# ==========================================
def train_and_validate():
    print(f"Loading Data...")
    train_dataset = FusionDataset(Config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataset = FusionDataset(Config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 1. 使用新模型
    print("Initializing AttentionFusionNet...")
    model = AttentionFusionNet(spectral_bands=Config.NUM_BANDS).to(Config.DEVICE)

    # 2. 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 3. 使用混合损失
    criterion = HybridLoss().to(Config.DEVICE)

    history = {'train_loss': [], 'val_loss': [], 'psnr': [], 'ssim': [], 'sam': [], 'ergas': []}

    print("=== Start Training with Attention & Gradient Loss ===")
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0
        for pan, mul, gt in train_loader:
            pan, mul, gt = pan.to(Config.DEVICE), mul.to(Config.DEVICE), gt.to(Config.DEVICE)
            optimizer.zero_grad()
            output = model(pan, mul)
            loss = criterion(output, gt)
            loss.backward()

            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        # 更新学习率
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss, m_psnr, m_ssim, m_sam, m_ergas = 0, 0, 0, 0, 0

        with torch.no_grad():
            for pan, mul, gt in val_loader:
                pan, mul, gt = pan.to(Config.DEVICE), mul.to(Config.DEVICE), gt.to(Config.DEVICE)
                output = model(pan, mul)
                loss = criterion(output, gt)
                val_loss += loss.item()

                out_np = np.clip(output.squeeze(0).cpu().numpy().transpose(1, 2, 0), 0, 1)
                gt_np = np.clip(gt.squeeze(0).cpu().numpy().transpose(1, 2, 0), 0, 1)

                m_psnr += Metrics.calculate_psnr(out_np, gt_np)
                m_ssim += Metrics.calculate_ssim(out_np, gt_np)
                m_sam += Metrics.calculate_sam(out_np, gt_np)
                m_ergas += Metrics.calculate_ergas(out_np, gt_np)

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = m_psnr / len(val_loader)

        # 记录所有指标
        history['val_loss'].append(avg_val_loss)
        history['psnr'].append(avg_psnr)
        history['ssim'].append(m_ssim / len(val_loader))
        history['sam'].append(m_sam / len(val_loader))
        history['ergas'].append(m_ergas / len(val_loader))

        print(f"Epoch [{epoch + 1}/{Config.EPOCHS}] LR:{optimizer.param_groups[0]['lr']:.6f} | "
              f"T-Loss: {avg_train_loss:.4f} | V-Loss: {avg_val_loss:.4f} | PSNR: {avg_psnr:.2f}")

    torch.save(model.state_dict(), os.path.join(Config.RESULT_DIR, 'attention_model.pth'))
    plot_metrics(history)  # 使用 v3 的绘图函数


# ==========================================
# 7. 绘图与推理
# ==========================================
def plot_metrics(history):
    epochs = range(1, Config.EPOCHS + 1)
    # --- 1.绘图 ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.title('Hybrid Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.PLOT_DIR, 'loss_curve.png'))
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(epochs, history['psnr'], 'g-')
    axs[0, 0].set_title('PSNR')
    axs[0, 1].plot(epochs, history['ssim'], 'b-')
    axs[0, 1].set_title('SSIM')
    axs[1, 0].plot(epochs, history['sam'], 'r-')
    axs[1, 0].set_title('SAM')
    axs[1, 1].plot(epochs, history['ergas'], 'm-')
    axs[1, 1].set_title('ERGAS')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOT_DIR, 'metrics.png'))
    plt.close()

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


def inference():
    print("Inference...")
    dataset = RealTestDataset(Config.TEST_PAN_DIR, Config.TEST_MS_DIR)
    model = AttentionFusionNet(spectral_bands=Config.NUM_BANDS).to(Config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(Config.RESULT_DIR, 'attention_model.pth')))
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            pan, mul, fname, max_val = dataset[i]
            pan = pan.unsqueeze(0).to(Config.DEVICE)
            mul = mul.unsqueeze(0).to(Config.DEVICE)
            fused = model(pan, mul)
            fused = np.clip(fused.squeeze(0).cpu().numpy() * max_val, 0, 65535).astype(np.uint16)
            tifffile.imwrite(os.path.join(Config.RESULT_DIR, f"Fused_Attn_{fname}"), np.transpose(fused, (1, 2, 0)))
            print(f"Saved: {fname}")


# ==========================================
# 8. 热力图
# ==========================================
def generate_report_assets(model_path, save_dir, sample_index=0, zoom_box=None):
    """
    生成对比图：
    1. 预测结果 vs Ground Truth
    2. 残差热力图 (Error Map)
    3. 局部放大图 (Zoom-in)

    参数:
        sample_index: 在验证集中选择第几张图进行分析
        zoom_box: 局部放大区域 (x, y, w, h)，如果为 None 则不生成放大图
    """
    print(f"正在生成汇报素材 (Index: {sample_index})...")

    # 1. 准备数据 (使用验证集，因为需要 Ground Truth)
    dataset = FusionDataset(Config.VAL_DIR, mode='val')
    if len(dataset) <= sample_index:
        print("错误: 索引超出了数据集大小")
        return

    pan, mul, gt = dataset[sample_index]

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # 2. 模型推理
    model = AttentionFusionNet(spectral_bands=Config.NUM_BANDS).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    with torch.no_grad():
        # 增加 batch 维度
        pan_t = pan.unsqueeze(0).to(Config.DEVICE)
        mul_t = mul.unsqueeze(0).to(Config.DEVICE)

        # 预测
        fused = model(pan_t, mul_t)

    # 3. 数据转换 (Tensor -> Numpy, CHW -> HWC)
    # 反归一化并转为 float 用于计算误差 (不转 uint16 以免丢失精度)
    pred_np = fused.squeeze(0).cpu().numpy().transpose(1, 2, 0) * Config.MAX_VAL
    gt_np = gt.cpu().numpy().transpose(1, 2, 0) * Config.MAX_VAL

    # 4. 计算残差热力图 (核心步骤)
    # Error = |Predicted - GT|
    # 对所有波段求平均，得到一张单通道的“综合误差图”
    # 也可以只看某个波段，例如: error_map = np.abs(pred_np[:,:,4] - gt_np[:,:,4]) (看红光波段)
    diff = np.abs(pred_np - gt_np)
    error_map = np.mean(diff, axis=2)

    # 为了让热力图对比更强烈，可以设置一个统一的上限 (vmax)
    # 比如误差超过 200 (在 2047 量级下) 就显示为最红
    heatmap_vmax = 0.1 * Config.MAX_VAL

    # 5. 绘图并保存
    os.makedirs(save_dir, exist_ok=True)

    # --- A. 保存完整热力图 ---
    plt.figure(figsize=(8, 6))
    plt.imshow(error_map, cmap='jet', vmin=0, vmax=heatmap_vmax)
    plt.colorbar(label='Absolute Error')
    plt.title(f'Error Map (Sample {sample_index})')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f'ErrorMap_Full_{sample_index}.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"已保存: ErrorMap_Full_{sample_index}.png")

    # --- B. 局部放大图 (Zoom-in) ---
    if zoom_box:
        x, y, w, h = zoom_box

        # 裁剪区域
        crop_pred = pred_np[y:y + h, x:x + w, :]
        crop_gt = gt_np[y:y + h, x:x + w, :]
        crop_error = error_map[y:y + h, x:x + w]

        # 显示真彩色 (假设波段 5-3-2 是 RGB，根据你的数据调整)
        # 归一化到 0-1 用于显示
        def norm_vis(img):
            vis = img[:, :, [4, 2, 1]]  # Band 5, 3, 2 (RGB) - indices 4, 2, 1
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            return vis

        # 1. 预测图局部
        plt.imsave(os.path.join(save_dir, f'Zoom_Pred_{sample_index}.png'), norm_vis(crop_pred))
        # 2. GT图局部
        plt.imsave(os.path.join(save_dir, f'Zoom_GT_{sample_index}.png'), norm_vis(crop_gt))
        # 3. 误差图局部
        plt.figure(figsize=(4, 4))
        plt.imshow(crop_error, cmap='jet', vmin=0, vmax=heatmap_vmax)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'Zoom_Error_{sample_index}.png'), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"已保存局部放大图 (位置: {zoom_box})")


if __name__ == '__main__':
    # 1. 训练 (如果已训练可注释)
    train_and_validate()

    # 2. 常规推理(如果已推理可注释)
    # inference()

    # 3. 热力图
    # model_path = os.path.join(Config.RESULT_DIR, 'attention_model.pth')
    # heatmap_dir = os.path.join(Config.ROOT_DIR, '汇报素材_Attention')
    # if not os.path.exists(heatmap_dir): os.makedirs(heatmap_dir)
    # for i in range(10, 20):
    #     # 运行
    #     generate_report_assets(
    #         model_path=model_path,
    #         save_dir=heatmap_dir,
    #         sample_index=i,  # 选择验证集中第3张图 (多试几个索引找到典型的)
    #         zoom_box=None
    #     )
