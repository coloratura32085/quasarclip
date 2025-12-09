import os, warnings, pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from astropy.wcs import WCS
from PIL import Image
from tqdm import tqdm

# --------------- 基础工具 ----------------- #
def crop_image(image, center, crop_size=(224, 224)):
    """中心 (x,y) 为旋转坐标，支持越界自动补黑边"""
    w, h     = image.size
    cw, ch   = crop_size
    left     = max(0, center[0] - cw // 2)
    upper    = max(0, center[1] - ch // 2)
    right    = min(w, center[0] + cw // 2)
    lower    = min(h, center[1] + ch // 2)

    crop     = image.crop((left, upper, right, lower))
    canvas   = Image.new("RGB", crop_size, (0, 0, 0))
    paste_x  = max(0, (cw // 2) - center[0])
    paste_y  = max(0, (ch // 2) - center[1])
    canvas.paste(crop, (paste_x, paste_y))
    return canvas

# --------------- 单条任务 ----------------- #
def _process_one(it, base_path, cutout_size, save_dir, meta):
    """线程工作函数：成功保存图像或忽略失败"""
    objID, img_path, header = it
    row = meta.get(objID)
    if row is None:
        return  # 找不到元数据，忽略

    band = "irg"
    run_no_0 = f"{int(row['run'])}"
    run      = f"{int(row['run']):06d}"
    camcol   = f"{int(row['camcol'])}"
    field    = f"{int(row['field']):04d}"

    ra, dec  = row["ra"], row["dec"]
    rgb_name = f"frame-{band}-{run}-{camcol}-{field}.jpg"
    wcs      = WCS(header)
    naxis1, naxis2 = header["NAXIS1"], header["NAXIS2"]

    # 像素坐标 (origin=1)
    x, y = wcs.all_world2pix(ra, dec, 1)
    if ("RA" in header["CTYPE1"] and (x < 0 or x > naxis1 or y < 0 or y > naxis2)) or \
       ("RA" in header["CTYPE2"] and (x < 0 or x > naxis2 or y < 0 or y > naxis1)):
        return  # 坐标越界，忽略

    x, y = int(x), int(naxis2 - int(y))
    pci_file = os.path.join(base_path, run_no_0, rgb_name)
    if not os.path.exists(pci_file):
        return  # 图像文件缺失，忽略

    # 裁剪并保存
    try:
        img = Image.open(pci_file)
        crop = crop_image(img, (x, y), crop_size=(cutout_size, cutout_size))
        os.makedirs(save_dir, exist_ok=True)
        crop.save(os.path.join(save_dir, f"{objID}.jpg"), "JPEG")
    except Exception:
        return  # 任何异常，忽略

# --------------- 并行调度 ----------------- #
def process_images_multithread(items, base_path, cutout_size, save_dir, meta_df, num_threads=8):
    warnings.simplefilter("ignore")
    meta_dict = {str(r["objID"]): r for _, r in meta_df.iterrows()}  # objID → row

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(_process_one, it, base_path, cutout_size,
                               save_dir, meta_dict): it[0] for it in items}

        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="Processing images (threads)"):
            pass  # 不需要处理结果

# --------------- 主流程 ----------------- #
if __name__ == "__main__":
    with open("D:/database/tmp/headers_filtered.pkl", "rb") as f:
        items = pickle.load(f)

    base_path   = "F:/database/sourcedata/image/image_301"
    csv_path    = "D:/database/final_index.csv"
    CUTOUT_SIZE = 224
    SAVE_DIR    = "F:/database/sourcedata/rgb_224"
    NUM_THREADS = 6

    df = pd.read_csv(csv_path, dtype={"objID": str})
    process_images_multithread(
        items, base_path, CUTOUT_SIZE, SAVE_DIR, df, num_threads=NUM_THREADS
    )