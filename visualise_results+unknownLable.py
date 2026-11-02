"""
GHIST — 分割结果可视化脚本


    python visualise_results.py --experiment demo --epoch 3
    python visualise_results.py --experiment_root /data2/clzhang/weight \
        --experiment fold1_2026_10_21_19_34_51 --epoch 100

细胞类型图除了 config 里的 9 类颜色外，还会画出一类 Unknown（黑色）：

"""

import argparse
import os
import sys
import itertools
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch

# cell type colours, kept in sync with split_cell_types.py COLORS
CELL_TYPE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# 不属于上面 9 类（或没有预测结果）的细胞统一归为 Unknown，颜色黑色
UNKNOWN_CELL_TYPE = "Unknown"
UNKNOWN_CELL_TYPE_COLOR = "#000000"


def build_celltype_colormap(n_types):
    """构造离散颜色表（code 0 = 背景白，code i = 第 i 类，code n+1 = Unknown 黑）"""
    # code 0 is background (white); code i maps to CELL_TYPE_COLORS[i-1]
    palette = list(itertools.islice(itertools.cycle(CELL_TYPE_COLORS), n_types))
    cmap = ListedColormap(['#ffffff'] + palette + [UNKNOWN_CELL_TYPE_COLOR])
    cmap.set_bad(color='white')
    norm = BoundaryNorm(np.arange(-0.5, n_types + 2.5, 1.0), n_types + 2)
    return cmap, norm


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_processing"))
from tiff_io import imread_tiff


def get_experiment_path(experiment_root, experiment):
    """实验目录，支持 --experiment_root 指向仓库外（如 /data2/clzhang/weight）"""
    return os.path.join(experiment_root, experiment)


def load_experiment_config(experiment_root, experiment):
    """加载实验配置（train.py 以原文件名拷贝配置，因此匹配 config*.json）"""
    import glob
    import json

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

    print(f"[WARNING] 未找到配置文件: {exp_path}/config*.json")
    return None


def resolve_data_file(config, key, exp_path, fallback):
    """优先用配置里的 data_sources_predict 路径，找不到再回退"""
    fp = (config or {}).get("data_sources_predict", {}).get(key)
    if fp and os.path.exists(fp):
        return fp
    fp = os.path.join(exp_path, fallback)
    if os.path.exists(fp):
        return fp
    return None


def load_nuclei_seg(experiment_root, experiment, config=None, fp_override=None):
    """加载核分割图"""
    exp_path = get_experiment_path(experiment_root, experiment)
    fp = fp_override or resolve_data_file(
        config, "fp_nuc_seg", exp_path, "val_output/nuclei_seg.tif"
    )
    if fp is None:
        fp = "outs_dir/he_image_nuclei_seg.tif"
    if not os.path.exists(fp):
        print(f"[ERROR] 找不到核分割图，请检查路径")
        sys.exit(1)

    print(f"[INFO] 加载核分割图: {fp}")
    seg = tifffile.imread(fp)
    print(f"       形状: {seg.shape}, 数据类型: {seg.dtype}")
    return seg


def load_he_image(experiment_root, experiment, config=None, fp_override=None):
    """加载 H&E 图像"""
    exp_path = get_experiment_path(experiment_root, experiment)
    fp = fp_override or resolve_data_file(
        config, "fp_hist", exp_path, "val_output/he_image.tif"
    )
    if fp is None or not os.path.exists(fp):
        print(f"[WARNING] 找不到 H&E 图像: {fp}")
        return None
    print(f"[INFO] 加载 H&E 图像: {fp}")
    img = imread_tiff(fp)
    if img.ndim == 5:
        # (1, 1, H, W, C) -> (H, W, C)
        img = img[0, 0, ...]
    elif img.ndim == 4:
        img = img[0, ...]
    elif img.ndim == 2:
        # 灰度图转 RGB
        img = np.stack([img] * 3, axis=-1)
    print(f"       形状: {img.shape}")
    return img


def load_predicted_expr(experiment_root, experiment, epoch):
    """加载预测的基因表达 CSV"""
    exp_path = get_experiment_path(experiment_root, experiment)
    fp = f"{exp_path}/val_output/epoch_{epoch}_expr.csv"
    if not os.path.exists(fp):
        fp = f"{exp_path}/predict_output/epoch_{epoch}_expr.csv"
    if not os.path.exists(fp):
        # demo_predict mode writes epoch_demo_predict_expr.csv
        fp = f"{exp_path}/predict_output/epoch_demo_predict_expr.csv"
    if not os.path.exists(fp):
        print(f"[ERROR] 找不到预测表达文件: {fp}")
        sys.exit(1)

    print(f"[INFO] 加载预测表达: {fp}")
    df = pd.read_csv(fp, index_col=0)
    print(f"       细胞数: {len(df)}, 基因数: {len(df.columns)}")
    return df


def load_cell_types(config, experiment_root="experiments", experiment=None, epoch=None):
    """Load cell type labels"""
    if config is None:
        return None, None

    # predict mode: prefer cell types saved by inference
    if experiment is not None:
        exp_path = get_experiment_path(experiment_root, experiment)
        for fp_cand in [
            f"{exp_path}/predict_output/epoch_{epoch}_celltype.csv",
            f"{exp_path}/predict_output/epoch_demo_predict_celltype.csv",
            f"{exp_path}/val_output/epoch_{epoch}_celltype.csv",
        ]:
            if os.path.exists(fp_cand):
                print(f"[INFO] Loading predicted cell types: {fp_cand}")
                df = pd.read_csv(fp_cand, index_col=0)
                cell_types_ordered = config.get("data", {}).get("cell_types", [])
                return df, cell_types_ordered

    fp = config.get("data_sources_train_val", {}).get("fp_cell_type", "data_demo/celltype_filtered.csv")
    if not os.path.exists(fp):
        fp = "data_demo/celltype_filtered.csv"
    if not os.path.exists(fp):
        return None, None
    df = pd.read_csv(fp, index_col=0)
    cell_types_ordered = config.get("data", {}).get("cell_types", [])
    return df, cell_types_ordered


def build_spatial_map(nuclei_seg, df_expr, gene):
    """将基因表达值映射到核分割图的空间位置"""
    if gene not in df_expr.columns:
        print(f"[ERROR] 基因 '{gene}' 不在预测结果中")
        print(f"        可选基因示例: {list(df_expr.columns[:5])}...")
        sys.exit(1)

    cell_ids = df_expr.index.values.astype(np.float64)
    expr_values = df_expr[gene].values.astype(np.float32)

    seg_ids, inverse = np.unique(nuclei_seg, return_inverse=True)
    lut = np.zeros(len(seg_ids), dtype=np.float32)
    pos = np.searchsorted(seg_ids.astype(np.float64), cell_ids)
    pos = np.clip(pos, 0, max(len(seg_ids) - 1, 0))
    hit = seg_ids.astype(np.float64)[pos] == cell_ids
    lut[pos[hit]] = expr_values[hit]

    return lut[inverse].reshape(nuclei_seg.shape)


def build_celltype_map(nuclei_seg, df_ct, cell_types_ordered):
    """将细胞类型映射到核分割图的空间位置。

    编码规则:
        0              背景（白色）
        1..n           已知细胞类型（CELL_TYPE_COLORS[i-1]）
        n+1            Unknown（黑色）

    分割图里有核、但 CSV 中没有预测结果的细胞（被最小面积/表达数据过滤掉，
    或重叠区域去重时被丢弃），以及预测标签不在已知类型里的细胞，统一编码为
    Unknown，绘图时为黑色，而不是留白。
    """
    if df_ct is None:
        return None, None

    id_to_ct = dict(zip(df_ct.index, df_ct["ct"]))

    # 构建编码映射
    if cell_types_ordered:
        cell_types_ordered = list(cell_types_ordered)
    else:
        cell_types_ordered = list(df_ct["ct"].unique())

    ct_to_code = {str(ct).strip(): i + 1 for i, ct in enumerate(cell_types_ordered)}
    unknown_code = len(cell_types_ordered) + 1

    seg_ids, inverse = np.unique(nuclei_seg, return_inverse=True)
    lut = np.zeros(len(seg_ids), dtype=np.uint8)
    # 分割图里的每个细胞核先标记为 Unknown，之后有明确预测再覆盖
    lut[seg_ids != 0] = unknown_code
    for cell_id, ct in id_to_ct.items():
        code = ct_to_code.get(str(ct).strip())
        if code is None:
            continue
        pos = int(np.searchsorted(seg_ids, cell_id))
        if pos < len(seg_ids) and seg_ids[pos] == cell_id:
            lut[pos] = code

    n_unknown = int(np.count_nonzero(lut == unknown_code))
    if n_unknown:
        print(
            f"[INFO] {n_unknown} 个细胞核没有预测类型，标记为 "
            f"{UNKNOWN_CELL_TYPE} (黑色)"
        )

    return lut[inverse].reshape(nuclei_seg.shape), cell_types_ordered


def plot_gene_expression(he_img, expr_map, gene, output_path, roi=None):
    """绘制基因表达空间分布图"""
    if roi:
        h1, w1, h2, w2 = roi
        he_roi = he_img[h1:h2, w1:w2] if he_img is not None else None
        expr_roi = expr_map[h1:h2, w1:w2]
    else:
        he_roi = he_img
        expr_roi = expr_map

    if he_roi is not None:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes = [axes[0], axes[1], None]

    # H&E
    if he_roi is not None:
        axes[0].imshow(he_roi)
        axes[0].set_title("H&E Image", fontsize=14)
        axes[0].axis("off")

    # 基因表达热图
    masked = np.ma.masked_where(expr_roi == 0, expr_roi)
    vmax = np.percentile(expr_roi[expr_roi > 0], 95) if expr_roi.max() > 0 else 1
    im = axes[1].imshow(expr_roi, cmap="inferno",
                        norm=Normalize(vmin=0, vmax=vmax))
    axes[1].set_title(f"{gene} Expression (raw)", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 叠加
    if he_roi is not None:
        axes[2].imshow(he_roi, alpha=0.3)
        axes[2].imshow(masked, cmap="viridis", alpha=0.7,
                       norm=Normalize(vmin=0, vmax=vmax))
        axes[2].set_title(f"{gene} Overlay", fontsize=14)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


def plot_cell_types(he_img, cell_type_map, cell_types_ordered, output_path, roi=None):
    """绘制细胞类型空间分布图"""
    if cell_type_map is None:
        return

    if roi:
        h1, w1, h2, w2 = roi
        he_roi = he_img[h1:h2, w1:w2] if he_img is not None else None
        ct_roi = cell_type_map[h1:h2, w1:w2]
    else:
        he_roi = he_img
        ct_roi = cell_type_map

    if he_roi is not None:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes = [axes[0], axes[1], None]

    if he_roi is not None:
        axes[0].imshow(he_roi)
        axes[0].set_title("H&E Image", fontsize=14)
        axes[0].axis("off")

    # 细胞类型图（背景 code 0 显示为白色，Unknown code n+1 显示为黑色）
    n_types = len(cell_types_ordered)
    cmap, norm = build_celltype_colormap(n_types)
    ct_img = np.ma.masked_where(ct_roi == 0, ct_roi)
    im = axes[1].imshow(ct_img, cmap=cmap, norm=norm)
    axes[1].set_title("Cell Types", fontsize=14)
    axes[1].set_facecolor("white")
    axes[1].axis("off")

    # 叠加图
    if he_roi is not None:
        masked_ct = np.ma.masked_where(ct_roi == 0, ct_roi)
        axes[2].imshow(he_roi, alpha=0.3)
        axes[2].imshow(masked_ct, cmap=cmap, norm=norm, alpha=0.7)
        axes[2].set_title("Cell Types Overlay", fontsize=14)
        axes[2].axis("off")

    # 统一图例
    legend_elements = [
        Patch(facecolor=cmap(i + 1), label=ct)
        for i, ct in enumerate(cell_types_ordered)
    ]
    # Unknown 始终出现在图例里，即使当前 ROI 内没有该类型
    legend_elements.append(
        Patch(facecolor=UNKNOWN_CELL_TYPE_COLOR, label=UNKNOWN_CELL_TYPE)
    )
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=min(len(legend_elements), 5), fontsize=10)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GHIST 分割结果可视化")
    parser.add_argument("--experiment", default="demo", help="实验名称 (默认: demo)")
    parser.add_argument("--experiment_root", default="experiments",
                        help="存放实验文件夹的根目录，如 /data2/clzhang/weight")
    parser.add_argument("--epoch", default=3, type=int, help="epoch 编号 (默认: 3)")
    parser.add_argument("--gene", default="CD8A", help="要可视化的基因名 (默认: CD8A)")
    parser.add_argument("--genes_file", default=None, help="基因列表文件，每行一个基因")
    parser.add_argument("--roi", nargs=4, type=int, default=None,
                        help="ROI: x1 y1 x2 y2（像素坐标）")
    parser.add_argument("--no_he", action="store_true", help="不叠加 H&E 背景")
    parser.add_argument("--fp_hist", default=None,
                        help="H&E 图像路径（覆盖配置里的 fp_hist）")
    parser.add_argument("--fp_nuc_seg", default=None,
                        help="核分割图路径（覆盖配置里的 fp_nuc_seg，需与推理时一致）")
    parser.add_argument("--output_dir", default=None, help="输出目录")

    # 解析参数
    # 如果手动传入则用，否则用 sys.argv
    args = parser.parse_args()

    # 加载配置
    config = load_experiment_config(args.experiment_root, args.experiment)
    exp_path = get_experiment_path(args.experiment_root, args.experiment)

    # 输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{exp_path}/visualisations"
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    nuclei_seg = load_nuclei_seg(
        args.experiment_root, args.experiment, config, args.fp_nuc_seg
    )
    he_img = (
        None
        if args.no_he
        else load_he_image(args.experiment_root, args.experiment, config, args.fp_hist)
    )
    df_expr = load_predicted_expr(args.experiment_root, args.experiment, args.epoch)
    df_ct, cell_types_ordered = load_cell_types(
        config, args.experiment_root, args.experiment, args.epoch
    )

    # 确定要可视化的基因列表
    if args.genes_file and os.path.exists(args.genes_file):
        with open(args.genes_file) as f:
            genes = [line.strip() for line in f if line.strip()]
    else:
        genes = [args.gene]

    # 可视化每个基因
    for gene in genes:
        print(f"\n{'='*60}")
        print(f"[INFO] 可视化基因: {gene}")
        print(f"{'='*60}")

        expr_map = build_spatial_map(nuclei_seg, df_expr, gene)

        # 如果 ROI 存在，先裁剪核分割图和表达图再处理
        output_suffix = f"{gene}"
        if args.roi:
            output_suffix += f"_roi_{args.roi[0]}_{args.roi[1]}_{args.roi[2]}_{args.roi[3]}"

        plot_gene_expression(
            he_img, expr_map, gene,
            f"{output_dir}/{output_suffix}.png",
            roi=args.roi,
        )

        # 每个基因同时也保存细胞类型图（只生成一次，放入单独子文件夹）
        ct_dir = f"{output_dir}/cell_types"
        os.makedirs(ct_dir, exist_ok=True)
        if df_ct is not None and not os.path.exists(f"{ct_dir}/cell_types.png"):
            print(f"\n[INFO] 生成细胞类型空间分布图")
            ct_map, ct_ordered = build_celltype_map(nuclei_seg, df_ct, cell_types_ordered)
            plot_cell_types(
                he_img, ct_map, ct_ordered,
                f"{ct_dir}/cell_types.png",
                roi=args.roi,
            )

    # 统计信息
    print(f"\n{'='*60}")
    print(f"[DONE] 所有可视化结果已保存至: {output_dir}")
    print(f"{'='*60}")
    print(f"\n生成的文件:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
