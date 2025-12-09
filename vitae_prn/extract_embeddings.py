import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_from_disk
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# --- 导入你的自定义模块 ---
from root_path import ROOT_PATH
from models.clip_resnet18 import AstroClipModel
# 导入数据增强（必须与训练时一致）
from imageutil.trans import CustomCenterCrop


# --- 定义与训练时一致的 Transform ---
class FlattenAndReshape:
    def __call__(self, x):
        # 假设输入是 tensor 或 numpy
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # 这里的逻辑必须和训练脚本完全一致
        original_shape = x.shape
        x_flat = x.flatten()
        x_reshaped = x_flat.view(32, 32, 5)
        x_final = x_reshaped.permute(2, 0, 1)  # (C, H, W)
        return x_final


class InferenceDataset(Dataset):
    """简单的推理用 Dataset，应用 Transform"""

    def __init__(self, hf_dataset, data_key, transform=None):
        self.dataset = hf_dataset
        self.data_key = data_key
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data = item[self.data_key]

        # 转换为 Tensor
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)

        return data


def get_embeddings(model, dataloader, device='cuda'):
    """使用 DataLoader 批量提取特征"""
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            # 根据模型类型调用不同的 forward 方法
            # 注意：AstroClipModel 的 forward 可能需要 input_type 参数
            # 这里假设我们需要提取的是 Image 特征，或者根据传入的 model wrapper 决定
            embeddings = model(batch)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def load_trained_model(ckpt_path, device='cuda'):
    """加载 PyTorch Lightning 训练出来的 Checkpoint"""
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')  # 先加载到 CPU
    state_dict = checkpoint['state_dict']

    # 1. 清洗 state_dict 的 key (去除 'model.' 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v  # 去掉 "model."
        else:
            new_state_dict[k] = v

    # 2. 初始化模型
    # 注意：spec_weight_path 在这里不需要真实路径，因为我们会用 state_dict 覆盖权重
    model = AstroClipModel(spec_weight_path=None)

    # 3. 加载权重 (使用 strict=False 以防有一些不匹配的参数，如 loss scale)
    keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model keys loaded. Missing: {keys.missing_keys}, Unexpected: {keys.unexpected_keys}")

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Extract Embeddings from Trained AstroCLIP")

    # 路径设置
    parser.add_argument('--train_data', type=str, default=f'../data/data_g3/train_dataset')
    parser.add_argument('--test_data', type=str, default=f'../data/data_g3/test_dataset')
    parser.add_argument('--output_dir', type=str, default=f'{ROOT_PATH}/outputs/embeddings')

    # 必须提供的参数：你的模型权重路径
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the .ckpt file")

    # 模式选择
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'spectrum'],
                        help="Extract 'image' or 'spectrum' embeddings")

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型
    full_model = load_trained_model(args.ckpt, args.device)

    # 定义特定的提取函数
    if args.mode == 'image':
        # 包装一下，让它只输出 image embedding
        extractor_model = lambda x: full_model(x, input_type='image')
        data_key = 'image'
        # 图像预处理 (必须与训练一致!)
        transform = transforms.Compose([
            CustomCenterCrop(size=32),
            FlattenAndReshape(),
        ])
    elif args.mode == 'spectrum':
        extractor_model = lambda x: full_model(x, input_type='spectrum')
        data_key = 'spectrum'
        transform = None  # 光谱通常不需要 resize/reshape，如果训练时有处理，这里也要加

    # 2. 处理数据集
    datasets = {
        'test': args.test_data,
        'train': args.train_data
    }

    for split_name, data_path in datasets.items():
        print(f"\nProcessing {split_name} split from {data_path}...")

        # 加载 HuggingFace Dataset
        hf_dataset = load_from_disk(data_path)

        # 创建 PyTorch Dataset 和 Loader
        dataset = InferenceDataset(hf_dataset, data_key, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # 提取特征
        embeddings = get_embeddings(extractor_model, loader, device=args.device)

        # 准备保存数据
        save_dict = {
            'embeddings': embeddings,
        }

        # 如果有标签，也一起保存
        if 'z' in hf_dataset.column_names:
            save_dict['z'] = np.array(hf_dataset['z'])
        if 'params' in hf_dataset.column_names:
            save_dict['params'] = np.array(hf_dataset['params'])

        # 保存
        out_name = f"{split_name}_{args.mode}_embeddings.npz"
        out_path = os.path.join(args.output_dir, out_name)
        np.savez(out_path, **save_dict)
        print(f"Saved to: {out_path} | Shape: {embeddings.shape}")


if __name__ == '__main__':
    main()