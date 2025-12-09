from datasets import load_from_disk, concatenate_datasets
import pandas as pd
import numpy as np
import math
import os

feature = 3600
batch_size = 1000
spectrum_csv = "data_spectrum.csv"
redshift_csv = "redshift.csv"

def nom_flow_batch(data):
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return data / norm

def export_dataset(dataset, spectrum_csv, redshift_csv):
    if os.path.exists(spectrum_csv):
        os.remove(spectrum_csv)
    if os.path.exists(redshift_csv):
        os.remove(redshift_csv)

    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = dataset[start:end]

        spectra = np.array(batch["spectrum"])[:, :feature]
        spectra = nom_flow_batch(spectra)

        redshift = np.array(batch["z"])

        df_spectrum = pd.DataFrame(spectra)
        df_redshift = pd.DataFrame({"z": redshift})

        df_spectrum.to_csv(spectrum_csv, mode="a", header=not os.path.exists(spectrum_csv), index=False)
        df_redshift.to_csv(redshift_csv, mode="a", header=not os.path.exists(redshift_csv), index=False)

        print(f"✅ 已处理 {end}/{total} 条样本")

# ========= 主逻辑 =========
train_ds = load_from_disk('F:/database/filterWave/data_g3_z/train_dataset')
test_ds = load_from_disk('F:/database/filterWave/data_g3_z/test_dataset')

dataset = concatenate_datasets([train_ds, test_ds])

export_dataset(dataset, spectrum_csv, redshift_csv)
print("🎉 全部完成：已将 spectrum (前3600维, 归一化) 与 redshift 分别导出到 CSV。")
