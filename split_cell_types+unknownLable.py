"""GHIST - 把细胞类型语义分割图按类型拆分成单独 PNG

visualise_results.py 会把 9 类细胞类型画在一张图里（cell_types.png），
本脚本复用同样的数据与配色，但把每一类单独输出成一张图。

用法:
    python split_cell_types.py                          # demo 默认参数
    python split_cell_types.py --roi 1000 1000 2000 2000
    python split_cell_types.py --no_he                  # 只出纯掩码图
    python split_cell_types.py --he_file data_demo/he_image02.tif
    python split_cell_types.py --output_dir my_out

    # 推理结果（权重目录在仓库外）
    python split_cell_types.py \
        --experiment_root /data2/clzhang/weight \
        --experiment fold1_2026_10_21_19_34_51 \
        --epoch 100

每类默认生成两张图:
    <输出目录>/<类型>_mask.png     纯语义掩码（该类颜色 + 白色背景）
    <输出目录>/<类型>_overlay.png  叠加到 H&E 上的半透明掩码（无匹配 H&E 时自动跳过）

除了 9 类之外，还会额外输出一张 Unknown（黑色）掩码：
分割图里有核、但 CSV 中没有预测结果的细胞（被最小面积/表达数据过滤掉、
重叠区域去重时被丢弃），以及预测标签不在 9 类里的细胞，都归到这一类。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


CELL_TYPES_DEFAULT = [
    "B", "Myeloid", "Endothelial", "Fibroblast",
    "Macrophage", "Malignant", "Epithelial", "Plasma", "T",
]

# tab10 的颜色顺序，与 visualise_results.py 的配色保持一致
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# 不属于以上类型（或者分割图里有核但没有预测结果）的细胞统一归为 Unknown
UNKNOWN_CELL_TYPE = "Unknown"
UNKNOWN_CELL_TYPE_COLOR = "#000000"


def get_experiment_path(experiment_root, experiment):
    return os.path.join(experiment_root, experiment)


def load_experiment_config(experiment_root, experiment):
    exp_path = get_experiment_path(experiment_root, experiment)
    candidates = [
        os.path.join(exp_path, "config_demo.json"),
        os.path.join(exp_path, "config.json"),
    ]
    candidates += sorted(glob.glob(os.path.join(exp_path, "config*.json")))
    for fp in candidates:
        if os.path.exists(fp):
            print(f"[INFO] 加载配置文件: {fp}")
            with open(fp) as f:
                return json.load(f)
    return None


def resolve_data_file(config, key, exp_path, fallback):
    fp = (config or {}).get("data_sources_predict", {}).get(key)
    if fp and os.path.exists(fp):
        return fp
    fp = os.path.join(exp_path, fallback)
    if os.path.exists(fp):
        return fp
    return None


def load_nuclei_seg(experiment_root, experiment, config=None, fp_override=None):
    exp_path = get_experiment_path(experiment_root, experiment)
    fp = fp_override or resolve_data_file(
        config, "fp_nuc_seg", exp_path, "val_output/nuclei_seg.tif"
    )
    if fp is None:
        fp = "outs_dir/he_image_nuclei_seg.tif"
    if not os.path.exists(fp):
        print(f"[ERROR] 找不到核分割图: {fp}")
        sys.exit(1)
    print(f"[INFO] 加载核分割图: {fp}")
    seg = tifffile.imread(fp)
    # 分割图常以 float 存储 (label == 细胞 id), 查表索引前需转为整数
    if not np.issubdtype(seg.dtype, np.integer):
        print(f"[INFO] 分割图 dtype={seg.dtype}, 转为整数 (max label = {int(seg.max())})")
        seg = np.rint(seg).astype(np.int64)
    return seg


def read_as_rgb(fp):
    img = tifffile.imread(fp)
    if img.ndim == 5:
        img = img[0, 0, ...]
    elif img.ndim == 4:
        img = img[0, ...]
    elif img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] != 3 and img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    return img


def find_matching_he(seg_shape, data_demo=True, experiment_root="experiments",
                     experiment="demo", he_file=None, config=None):
    """找到与分割图尺寸一致的 H&E 图像；找不到返回 None。"""
    if he_file:
        if not os.path.exists(he_file):
            print(f"[WARNING] 指定的 H&E 文件不存在: {he_file}")
            return None
        img = read_as_rgb(he_file)
        print(f"[INFO] 加载 H&E 图像: {he_file}, 形状 {img.shape}")
        return img

    exp_path = get_experiment_path(experiment_root, experiment)
    candidates = []
    fp_cfg = resolve_data_file(config, "fp_hist", exp_path, "val_output/he_image.tif")
    if fp_cfg:
        candidates.append(fp_cfg)
    if data_demo:
        candidates += sorted(glob.glob("data_demo/he_image*.tif"))
    else:
        candidates.append(f"{exp_path}/val_output/he_image.tif")

    for fp in candidates:
        try:
            img = read_as_rgb(fp)
        except Exception:
            continue
        if img.shape[:2] == tuple(seg_shape):
            print(f"[INFO] H&E 与分割图尺寸一致, 使用: {fp}, 形状 {img.shape}")
            return img

    print("[WARNING] 未找到与分割图尺寸匹配的 H&E 图像, 跳过叠加图")
    return None


def build_celltype_map(nuclei_seg, df_ct, cell_types_ordered, matched_nuclei=None):
    """把细胞类型标签映射到核分割图的每个像素上，返回 uint8 编码图。

    编码规则: 0 = 背景, i+1 = cell_types_ordered[i], n+1 = Unknown(黑色)。
    有核但没有任何预测结果的细胞（例如被最小面积过滤掉、重叠去重时被丢弃）
    以及标签不在 cell_types_ordered 里的细胞，一律记为 Unknown。
    支持两种映射方式:
      1) 直接映射: 分割图里的标签就是 csv 的 c_id
      2) 间接映射: 分割图标签 k -> matched_nuclei 第 k 行 -> id_histology -> c_id
    """
    if not np.issubdtype(nuclei_seg.dtype, np.integer):
        nuclei_seg = np.rint(nuclei_seg).astype(np.int64)
    n_seg = int(nuclei_seg.max())
    cell_types_ordered = list(cell_types_ordered)
    ct_to_code = {str(ct).strip(): i + 1 for i, ct in enumerate(cell_types_ordered)}
    unknown_code = len(cell_types_ordered) + 1

    seg_ids = np.unique(nuclei_seg)
    seg_ids = seg_ids[seg_ids != 0]

    direct_ids = df_ct.index.values
    if np.intersect1d(seg_ids, direct_ids).size > 0:
        print("[INFO] 使用直接映射 (seg label == c_id)")
        lut = np.zeros(n_seg + 1, dtype=np.uint8)
        lut[1:] = unknown_code  # 默认 Unknown，有预测就覆盖
        codes = df_ct["ct"].astype(str).str.strip().map(ct_to_code).values
        valid = ~pd.isna(codes)
        lut[direct_ids[valid]] = codes[valid].astype(np.uint8)
        n_unknown = int(np.count_nonzero(lut[seg_ids] == unknown_code))
        print(f"[INFO] {n_unknown} 个细胞核没有预测类型, 记为 {UNKNOWN_CELL_TYPE}")
        return lut[nuclei_seg]

    if matched_nuclei is not None:
        print("[INFO] 使用间接映射 (seg label -> matched row -> id_histology)")
        id_to_ct = dict(zip(df_ct.index, df_ct["ct"]))
        lut = np.zeros(n_seg + 1, dtype=np.uint8)
        lut[1:] = unknown_code
        n_rows = min(n_seg + 1, len(matched_nuclei) + 1)
        for k in range(1, n_rows):
            hist_id = matched_nuclei.iloc[k - 1]["id_histology"]
            ct_str = id_to_ct.get(hist_id)
            code = ct_to_code.get(str(ct_str).strip()) if ct_str is not None else None
            if code is not None:
                lut[k] = code
        n_unknown = int(np.count_nonzero(lut[seg_ids] == unknown_code))
        print(f"[INFO] {n_unknown} 个细胞核没有预测类型, 记为 {UNKNOWN_CELL_TYPE}")
        return lut[nuclei_seg]

    print("[WARNING] 无法建立映射, 返回空图")
    return np.zeros(nuclei_seg.shape[:2], dtype=np.uint8)


def plot_single_type(mask_roi, ct_name, color, he_roi, output_path,
                     alpha=0.6, bg_alpha=0.3):
    """绘制单个细胞类型的掩码图（白底 + 该类颜色）及 H&E 叠加图。"""
    rgb = np.array(to_rgb(color))

    vis = np.full((*mask_roi.shape, 3), 1.0, dtype=np.float32)
    vis[mask_roi > 0] = rgb
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(vis)
    ax.set_title(ct_name, fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    if he_roi is not None:
        he = he_roi.astype(np.float32) / 255.0
        # bg_alpha 越小，H&E 背景越淡（向白色混合）
        he = he * bg_alpha + (1.0 - bg_alpha)
        m = (mask_roi > 0).astype(np.float32)[..., np.newaxis]
        blend = he * (1 - m * alpha) + rgb[np.newaxis, np.newaxis, :] * m * alpha
        blend = np.clip(blend, 0, 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(blend)
        ax.set_title(f"{ct_name} Overlay", fontsize=16)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path.replace("_mask.png", "_overlay.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="GHIST 按细胞类型拆分语义分割图")
    parser.add_argument("--experiment", default="demo", help="实验名称")
    parser.add_argument("--experiment_root", default="experiments",
                        help="存放实验文件夹的根目录，如 /data2/clzhang/weight")
    parser.add_argument("--epoch", default=None, type=int,
                        help="epoch 编号（仅用于默认输出目录命名）")
    parser.add_argument("--roi", nargs=4, type=int, default=None,
                        help="ROI: h1 w1 h2 w2 (像素坐标)")
    parser.add_argument("--no_he", action="store_true", help="不生成 H&E 叠加图")
    parser.add_argument("--he_file", default=None,
                        help="指定 H&E 图像路径（默认自动匹配分割图尺寸）")
    parser.add_argument("--fp_hist", default=None,
                        help="H&E 图像路径（覆盖配置里的 fp_hist）")
    parser.add_argument("--fp_nuc_seg", default=None,
                        help="核分割图路径（覆盖配置里的 fp_nuc_seg）")
    parser.add_argument("--fp_cell_type", default=None,
                        help="细胞类型 CSV 路径（覆盖配置里的 fp_cell_type）")
    parser.add_argument("--fp_matched_nuclei", default=None,
                        help="matched_nuclei CSV 路径（seg label -> c_id 映射用）")
    parser.add_argument("--output_dir", default=None, help="输出目录")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="掩码叠加到 H&E 上的不透明度 (0-1)")
    parser.add_argument("--bg_alpha", type=float, default=0.3,
                        help="H&E 背景浓度 (0-1)，越小背景越淡")
    args = parser.parse_args()

    config = load_experiment_config(args.experiment_root, args.experiment)
    exp_path = get_experiment_path(args.experiment_root, args.experiment)
    cell_types = (
        config.get("data", {}).get("cell_types", CELL_TYPES_DEFAULT)
        if config else CELL_TYPES_DEFAULT
    )
    data_demo = args.experiment == "demo"

    nuclei_seg = load_nuclei_seg(
        args.experiment_root, args.experiment, config, args.fp_nuc_seg
    )
    if args.no_he:
        he_img = None
    else:
        he_img = find_matching_he(
            nuclei_seg.shape[:2], data_demo=data_demo,
            experiment_root=args.experiment_root, experiment=args.experiment,
            he_file=args.fp_hist or args.he_file, config=config,
        )

    fp_ct = args.fp_cell_type
    if fp_ct is None and config:
        fp_ct = config.get("data_sources_train_val", {}).get("fp_cell_type")
    if fp_ct is None or not os.path.exists(fp_ct):
        for cand in ("data_demo/celltype_filtered.csv",
                     f"{exp_path}/predict_output/epoch_{args.epoch}_celltype.csv",
                     f"{exp_path}/val_output/epoch_{args.epoch}_celltype.csv"):
            if os.path.exists(cand):
                fp_ct = cand
                break
    if not fp_ct or not os.path.exists(fp_ct):
        print("[ERROR] 找不到细胞类型文件，请用 --fp_cell_type 指定")
        sys.exit(1)
    print(f"[INFO] 细胞类型文件: {fp_ct}")
    df_ct = pd.read_csv(fp_ct, index_col=0)
    print(f"[INFO] 细胞类型数据: {len(df_ct)} cells, {len(cell_types)} types")

    # matched_nuclei 提供 seg label -> c_id 的桥梁（无则只用直接映射）
    fp_mn = args.fp_matched_nuclei
    if fp_mn is None:
        for cand in ("data_demo/matched_nuclei_filtered.csv",
                     f"{exp_path}/matched_nuclei_filtered.csv"):
            if os.path.exists(cand):
                fp_mn = cand
                break
    df_mn = pd.read_csv(fp_mn) if fp_mn and os.path.exists(fp_mn) else None

    ct_map = build_celltype_map(nuclei_seg, df_ct, cell_types, df_mn)

    if args.roi:
        h1, w1, h2, w2 = args.roi
        ct_roi = ct_map[h1:h2, w1:w2]
        he_roi = he_img[h1:h2, w1:w2] if he_img is not None else None
    else:
        ct_roi = ct_map
        he_roi = he_img

    if args.output_dir:
        output_dir = args.output_dir
    elif args.epoch is not None:
        output_dir = f"{exp_path}/visualisations/cell_types_split/epoch_{args.epoch}"
    else:
        output_dir = f"{exp_path}/visualisations/cell_types_split"
    os.makedirs(output_dir, exist_ok=True)

    for i, ct_name in enumerate(cell_types):
        code = i + 1
        mask_roi = (ct_roi == code).astype(np.uint8)
        n_px = int(mask_roi.sum())
        extra = " / %s_overlay.png" % ct_name if he_roi is not None else ""
        print(f"  [{ct_name}] {n_px} px -> {ct_name}_mask.png{extra}")
        plot_single_type(mask_roi, ct_name, COLORS[i], he_roi,
                         os.path.join(output_dir, f"{ct_name}_mask.png"),
                         alpha=args.alpha, bg_alpha=args.bg_alpha)

    # Unknown（有核但无预测 / 不在 9 类里）单独出一张黑色掩码图
    unknown_code = len(cell_types) + 1
    mask_roi = (ct_roi == unknown_code).astype(np.uint8)
    n_px = int(mask_roi.sum())
    extra = " / %s_overlay.png" % UNKNOWN_CELL_TYPE if he_roi is not None else ""
    print(f"  [{UNKNOWN_CELL_TYPE}] {n_px} px -> {UNKNOWN_CELL_TYPE}_mask.png{extra}")
    plot_single_type(mask_roi, UNKNOWN_CELL_TYPE, UNKNOWN_CELL_TYPE_COLOR, he_roi,
                     os.path.join(output_dir, f"{UNKNOWN_CELL_TYPE}_mask.png"),
                     alpha=args.alpha, bg_alpha=args.bg_alpha)

    print(f"\n[DONE] 已保存至: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, f)
        print(f"  {f}  ({os.path.getsize(fp) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
