import argparse

import os

import sys

import numpy as np

import matplotlib.pyplot as plt

import copy  # 用于深拷贝模型权重



# PyTorch 相关

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset



# Sklearn 相关

from sklearn.neighbors import KNeighborsRegressor

from sklearn.manifold import TSNE

from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler



# 导入你的路径配置

from root_path import ROOT_PATH



# 自动检测设备

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ==========================================

# Part 1: 模型定义 (MLP & Training Loop)

# ==========================================



class MLP(nn.Sequential):

    """简单的多层感知机 (MLP) 模型结构"""

    def __init__(self, n_in, n_out, n_hidden=(256, 128, 64), act=None, dropout=0.1):

        if act is None:

            act = [nn.LeakyReLU()] * (len(n_hidden) + 1)

        

        layers = []

        n_ = [n_in, *n_hidden, n_out]

        

        for i in range(len(n_) - 2):

            layers.append(nn.Linear(n_[i], n_[i + 1]))

            layers.append(act[i])

            layers.append(nn.Dropout(p=dropout))

        

        layers.append(nn.Linear(n_[-2], n_[-1]))

        super(MLP, self).__init__(*layers)



def few_shot_mlp(

    X_train: np.ndarray,

    y_train: np.ndarray,

    X_test: np.ndarray,

    y_test: np.ndarray,  # 【修改1】新增参数：传入测试集标签用于监控

    output_dir: str,

    max_epochs: int = 100,

    lr: float = 1e-3,

    batch_size: int = 64

) -> np.ndarray:

    """

    使用 PyTorch 训练 MLP 回归模型

    【逻辑修改】：每个 Epoch 结束后计算 Test Loss，并保存 Test Loss 最低时的模型权重。

    """

    print(f"  -> Training MLP on {device} (Epochs: {max_epochs})...")

    

    # 设置随机种子

    torch.manual_seed(42)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(42)



    # 1. 准备 Train 数据

    train_dataset = TensorDataset(

        torch.tensor(X_train, dtype=torch.float32),

        torch.tensor(y_train, dtype=torch.float32),

    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



    # 2. 准备 Test 数据 (用于监控 Monitor)

    # 注意：这里我们创建 DataLoader 是为了方便分批计算 Loss，防止爆显存

    test_dataset = TensorDataset(

        torch.tensor(X_test, dtype=torch.float32),

        torch.tensor(y_test, dtype=torch.float32)

    )

    # 测试集 batch_size 可以大一点，因为不需要反向传播

    test_monitor_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)



    # 3. 初始化模型

    num_features = 1 if len(y_train.shape) == 1 else y_train.shape[1]

    model = MLP(n_in=X_train.shape[1], n_out=num_features)

    model.to(device)



    # 4. 优化器和损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.L1Loss() # MAE Loss



    # 5. 训练循环变量

    best_test_loss = float('inf')

    best_model_weights = None # 用于在内存中暂存最佳权重

    best_epoch = 0



    for epoch in range(max_epochs):

        # --- Training Step ---

        model.train()

        train_loss_sum = 0.0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            if labels.ndim == 1: labels = labels.unsqueeze(1)



            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss_sum += loss.item()

        

        avg_train_loss = train_loss_sum / len(train_loader)



        # --- Monitoring Step (Calculate Test Loss) ---

        model.eval()

        test_loss_sum = 0.0

        with torch.no_grad():

            for inputs, labels in test_monitor_loader:

                inputs, labels = inputs.to(device), labels.to(device)

                if labels.ndim == 1: labels = labels.unsqueeze(1)

                

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                test_loss_sum += loss.item()

        

        avg_test_loss = test_loss_sum / len(test_monitor_loader)



        # --- Check Best (Monitor Test Loss) ---

        if avg_test_loss < best_test_loss:

            best_test_loss = avg_test_loss

            best_epoch = epoch

            # 深拷贝当前模型权重到内存

            best_model_weights = copy.deepcopy(model.state_dict())



        # 打印日志 (每20轮)

        if (epoch + 1) % 20 == 0:

            print(f"     Epoch [{epoch+1}/{max_epochs}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} {'(*)' if epoch==best_epoch else ''}")



    # 6. 训练结束，加载历史上最好的权重

    print(f"  -> Loading Best Model from Epoch {best_epoch+1} (Test Loss: {best_test_loss:.4f})")

    model.load_state_dict(best_model_weights)

    

    # 保存最佳模型到硬盘 (可选)

    torch.save(model.state_dict(), os.path.join(output_dir, "best_mlp_model.pth"))



    # 7. 最终预测 (使用最佳权重)

    model.eval()

    all_preds = []

    

    # 重新创建一个 loader 用于纯预测 (只取 X)

    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    pred_loader = DataLoader(TensorDataset(test_tensor), batch_size=2048, shuffle=False)

    

    with torch.no_grad():

        for (batch_x,) in pred_loader:

            batch_x = batch_x.to(device)

            preds = model(batch_x)

            all_preds.append(preds.cpu().numpy())

            

    return np.concatenate(all_preds).flatten()



def zero_shot_knn(

    X_train: np.ndarray, 

    y_train: np.ndarray, 

    X_test: np.ndarray, 

    n_neighbors: int = 15

) -> np.ndarray:

    """

    使用 Sklearn KNN 进行回归

    """

    print(f"  -> Training KNN (k={n_neighbors})...")

    neigh = KNeighborsRegressor(weights="distance", n_neighbors=n_neighbors, n_jobs=-1)

    neigh.fit(X_train, y_train)

    return neigh.predict(X_test)



# ==========================================

# Part 2: 数据加载与工具函数

# ==========================================



def load_data(data_dir, mode):

    """加载 .npz 数据"""

    train_path = os.path.join(data_dir, f"train_{mode}_embeddings.npz")

    test_path = os.path.join(data_dir, f"test_{mode}_embeddings.npz")



    print(f"正在加载数据...\n  Train: {train_path}\n  Test:  {test_path}")



    try:

        train_data = np.load(train_path)

        test_data = np.load(test_path)

    except FileNotFoundError as e:

        print(f"\n❌ 错误: 找不到文件 {e.filename}。")

        sys.exit(1)



    if 'embeddings' not in train_data:

        print(f"❌ 错误: .npz 文件中找不到键 'embeddings'。")

        sys.exit(1)



    X_train = train_data['embeddings']

    y_train = train_data['z'] 

    X_test = test_data['embeddings']

    y_test = test_data['z']



    print(f"✅ 加载成功。训练集: {X_train.shape}, 测试集: {X_test.shape}")

    return X_train, y_train, X_test, y_test



def evaluate_and_plot(y_true, y_pred, model_title, output_dir, log_file):

    """评估指标 + 绘图"""

    r2 = r2_score(y_true, y_pred)

    delta_z = np.abs(y_pred - y_true) / (1 + y_true)

    

    frac_15 = np.mean(delta_z < 0.10) * 100

    frac_30 = np.mean(delta_z < 0.30) * 100

    sigma_nmad = 1.48 * np.median(delta_z)



    header = f"--- 评估结果: {model_title} ---"

    res_text = (

        f"  R² Score: {r2:.4f}\n"

        f"  Sigma_NMAD: {sigma_nmad:.4f}\n"

        f"  偏差 < 0.15 比例: {frac_15:.2f}%\n"

        f"  偏差 < 0.30 比例: {frac_30:.2f}%"

    )



    print(header)

    print(res_text)

    print("-" * 30)



    log_file.write(header + '\n')

    log_file.write(res_text + '\n\n')



    plt.figure(figsize=(8, 8))

    plot_indices = np.arange(len(y_true))

    if len(y_true) > 10000:

        plot_indices = np.random.choice(len(y_true), 10000, replace=False)
    

    plt.scatter(y_true[plot_indices], y_pred[plot_indices], alpha=0.3, s=5, c='blue')

    min_val = min(y_true.min(), y_pred.min())

    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal")



    plt.title(f"{model_title}\n$R^2={r2:.4f}, \sigma_{{NMAD}}={sigma_nmad:.4f}$")

    plt.xlabel("True Redshift")

    plt.ylabel("Predicted Redshift")

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()



    output_path = os.path.join(output_dir, f"{model_title}_scatter.png")

    plt.savefig(output_path, dpi=100)

    plt.close()



def visualize_with_tsne(embeddings, redshifts, model_name, output_dir):

    print("\n正在进行 t-SNE 降维可视化...")

    n_samples = 3000

    if len(embeddings) > n_samples:

        indices = np.random.choice(len(embeddings), size=n_samples, replace=False)

        embeddings_sample = embeddings[indices]

        redshifts_sample = redshifts[indices]

    else:

        embeddings_sample = embeddings

        redshifts_sample = redshifts



    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')

    embeddings_2d = tsne.fit_transform(embeddings_sample)



    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=redshifts_sample, cmap='Spectral_r', s=15, alpha=0.7)

    plt.title(f"t-SNE Visualization ({model_name})")

    plt.colorbar(scatter, label="Redshift")

    

    out_path = os.path.join(output_dir, f"{model_name}_tsne.png")

    plt.savefig(out_path, dpi=100)

    print(f"✅ t-SNE 保存至: {out_path}")

    plt.close()



# ==========================================

# Part 3: 主程序

# ==========================================



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Embeddings")
    
    default_emb_dir = os.path.join(ROOT_PATH, "outputs", "embeddings")
    default_res_dir = os.path.join(ROOT_PATH, "outputs", "results")

    parser.add_argument("--mode", type=str, default="image", choices=["image", "spectrum"], 
                        help="评估哪种嵌入 (image 或 spectrum)")
    parser.add_argument("--data_dir", type=str, default=default_emb_dir)
    parser.add_argument("--output_dir", type=str, default=default_res_dir)
    parser.add_argument("--epochs", type=int, default=100, help="MLP 训练轮数")
    
    # 添加cropsize参数
    parser.add_argument("--cropsize", type=int, default=32, help="Center crop size for images")
    # 添加自定义嵌套路径参数
    parser.add_argument("--subdir", type=str, default="", help="Additional subdirectory for custom organization")

    args = parser.parse_args()
    
    # 根据cropsize和实验参数创建特定的输出目录
    experiment_dir = f"cropsize_{args.cropsize}"
    output_dir = os.path.join(args.output_dir, experiment_dir)
    
    # 如果提供了自定义子目录，则添加到路径中
    if args.subdir:
        output_dir = os.path.join(output_dir, args.subdir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, "evaluation_log.txt")



    with open(log_file_path, 'a') as log_file:

        print(f"\n=== 开始评估: {args.mode} mode ===")

        log_file.write(f"\n=== Evaluation Run: {args.mode} (Epochs: {args.epochs}) ===\n")



        # 1. 加载

        X_train, y_train, X_test, y_test = load_data(args.data_dir, args.mode)



        # 2. 标准化 (Feature Scaling)

        scaler_X = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)

        X_test_scaled = scaler_X.transform(X_test)



        # 标签标准化 (Label Scaling)

        scaler_y = StandardScaler()

        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # 注意：这里我们还需要一个 y_test 的标准化版本，用于训练过程中的 loss 计算

        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()



        # ----------------------------
        # Run KNN (Zero-Shot)
        # ----------------------------
        print("\n[1/3] Running KNN...")
        y_pred_knn_scaled = zero_shot_knn(X_train_scaled, y_train_scaled, X_test_scaled, n_neighbors=20)

        y_pred_knn = scaler_y.inverse_transform(y_pred_knn_scaled.reshape(-1, 1)).flatten()

        evaluate_and_plot(y_test, y_pred_knn, f"{args.mode}_KNN", output_dir, log_file)

        # ----------------------------
        # Run MLP (Few-Shot)
        # ----------------------------
        print("\n[2/3] Running MLP...")
        # 【修改】这里传入了 y_test_scaled，让函数内部去监控 Test Loss
        y_pred_mlp_scaled = few_shot_mlp(
            X_train_scaled, 
            y_train_scaled, 
            X_test_scaled, 
            y_test_scaled, # <--- 传入测试集标签用于 Monitor
            output_dir=output_dir,
            max_epochs=args.epochs
        )
        
        y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).flatten()
        evaluate_and_plot(y_test, y_pred_mlp, f"{args.mode}_MLP", output_dir, log_file)

        # ----------------------------
        # Visualization
        # ----------------------------
        print("\n[3/3] Visualizing...")
        visualize_with_tsne(X_test_scaled, y_test, args.mode, output_dir)

    print(f"\n🎉 任务完成！结果已保存至: {output_dir}")
