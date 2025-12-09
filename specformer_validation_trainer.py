import lightning as L
from torch.utils.data import DataLoader
from datasets import load_from_disk
import argparse
from dataset_util.PairDataset import PairDataset
from models.spec_validation import AstroClipModel



# 定义命令行参数解析器
def get_args():
    parser = argparse.ArgumentParser(description='Train AstroClip Model')
    parser.add_argument('--train_data_path', default="", type=str, required=True, help='Path to training data')
    parser.add_argument('--test_data_path', default="", type=str, required=True, help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for optimizer')
    parser.add_argument('--spec_weight_path', type=str, required=True, help='Path to spectrum encoder weights')
    return parser.parse_args()


# 加载数据集并创建数据加载器
def create_dataloaders(args):
    train_dataset = load_from_disk(args.train_data_path)
    test_dataset = load_from_disk(args.test_data_path)

    # 创建自定义数据集实例
    train_dataset = PairDataset(train_dataset)
    test_dataset = PairDataset(test_dataset)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


# 定义 Trainer 和训练循环
def main():
    args = get_args()

    # 初始化模型
    model = AstroClipModel(
        spec_weight_path=args.spec_weight_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs
    )

    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(args)

    # 初始化 Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",  # 自动选择合适的加速器（CPU/GPU）
        # 自动选择可用的设备
        check_val_every_n_epoch=1,  # 每个epoch后进行验证
    )

    # 开始训练
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()