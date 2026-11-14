
"""修复 H&E <-> Xenium 配准，在 GHIST Step 4之后，产出正确的匹配表与表达矩阵。"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PXM = 0.2125          # Xenium 像素尺寸 (um)
S_MICRON = 0.2125     # he_image_nuclei_seg_microns.tif 相对原始 H&E 的索引系数

FP_XMAP = "output/xenium_cell_ids_dict.csv"
FP_BND = "nucleus_boundaries.csv.gz"
FP_HESEG = "output/he_image_nuclei_seg.tif"
FP_HEMICRON = "output/he_image_nuclei_seg_microns.tif"
FP_CELLS = "cells.csv.gz"
FP_ALIGN = "Xenium_V1_humanLung_Cancer_FFPE_he_imagealignment.csv"
FP_CGM = "output/cell_gene_matrix.csv"
FP_MORPH = "morphology_focus/morphology_focus_0000.ome.tif"

T0 = time.time()


def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)


# --------------------------------------------------------------------------
# 1. 正确的 Xenium 核分割图（1 um/px，完整画布）
# --------------------------------------------------------------------------
def build_xenium_mask(rh, rw):
    xmap = pd.read_csv(FP_XMAP, index_col=0)["cell_id_num"]
    nb = pd.read_csv(FP_BND)
    nb["num"] = nb["cell_id"].map(xmap)
    nb = nb.dropna(subset=["num"])
    log("boundaries: %d 行 / %d 个核" % (len(nb), nb["num"].nunique()))
    img = Image.new("I", (rw, rh), 0)
    draw = ImageDraw.Draw(img)
    for num, g in nb.groupby("num", sort=False):
        pts = list(zip(g["vertex_x"].tolist(), g["vertex_y"].tolist()))
        if len(pts) >= 3:
            draw.polygon(pts, fill=int(num))
    out = np.asarray(img, dtype=np.uint32)
    log("xenium mask %s, 标签 %d, 覆盖 %.4f"
        % (out.shape, len(np.unique(out)) - 1, float((out > 0).mean())))
    return out


# --------------------------------------------------------------------------
# 2. 把 H&E 核分割图重采样到 Xenium 坐标系
# --------------------------------------------------------------------------
def warp_he(seg, rh, rw, a3i, orig_rows, orig_cols, dy=0.0, dx=0.0,
            sc=1.0, idx_scale=1.0, chunk=256):
    """seg 是 ORIGINAL H&E 方向的大图；输出 Xenium 1um/px 帧。

    标定（已用 IoU 反解验证）：
        (x_um, y_um) --/PXM--> Xenium 像素 --inv(M)--> 原始 H&E (r, c)
        seg 行 = orig_cols-1-c,  seg 列 = orig_rows-1-r
    """
    cx = (orig_cols - 1) / 2.0
    cy = (orig_rows - 1) / 2.0
    out = np.zeros((rh, rw), np.uint32)
    for r0 in range(0, rh, chunk):
        r1 = min(r0 + chunk, rh)
        yy, xx = np.meshgrid(np.arange(r0, r1) + 0.5, np.arange(rw) + 0.5,
                             indexing="ij")
        xp = (xx.ravel() + dx) / PXM
        yp = (yy.ravel() + dy) / PXM
        c = a3i[0, 0] * xp + a3i[0, 1] * yp + a3i[0, 2]
        r = a3i[1, 0] * xp + a3i[1, 1] * yp + a3i[1, 2]
        c = cx + (c - cx) * sc
        r = cy + (r - cy) * sc
        ri = np.rint((orig_cols - 1 - c) * idx_scale).astype(np.int64)
        ci = np.rint((orig_rows - 1 - r) * idx_scale).astype(np.int64)
        ci = seg.shape[1] - 1 - ci        # rot90 列翻转（经过 IoU 反解验证）
        ok = (ri >= 0) & (ri < seg.shape[0]) & (ci >= 0) & (ci < seg.shape[1])
        v = np.zeros(ri.size, np.uint32)
        v[ok] = seg[ri[ok], ci[ok]]
        out[r0:r1] = v.reshape(r1 - r0, rw)
    return out


# --------------------------------------------------------------------------
# 3. Step 4 的原始匹配规则
# --------------------------------------------------------------------------
def match(xen_mask, he_mask, min_overlap_fraction):
    mx = int(max(xen_mask.max(), he_mask.max())) + 1
    m = (xen_mask > 0) & (he_mask > 0)
    xv = xen_mask[m].astype(np.int64)
    hv = he_mask[m].astype(np.int64)
    key = hv * 200000 + xv
    u, cnt = np.unique(key, return_counts=True)
    hid = (u // 200000).astype(np.int64)
    xid = (u % 200000).astype(np.int64)
    size = np.bincount(he_mask.ravel(), minlength=mx)
    df = pd.DataFrame({"id_histology": hid, "id_xenium": xid, "overlap": cnt})
    df["size_pix_histology"] = size[df["id_histology"].values]

    df_match = df.loc[df.groupby("id_histology")["overlap"].idxmax()]
    df_match = df_match[df_match["id_histology"] != 0]
    log("matched_nuclei: %d 对（H&E 核 %d 个，覆盖 Xenium 细胞 %d 个）"
        % (len(df_match), df_match.id_histology.nunique(),
           df_match.id_xenium.nunique()))

    keep = df_match[df_match["overlap"] >= min_overlap_fraction * df_match["size_pix_histology"]]
    keep = keep.sort_values(["id_xenium", "overlap"], ascending=[True, False])
    keep = keep.drop_duplicates(subset="id_xenium", keep="first")
    keep = keep.sort_values("id_histology")
    log("matched_nuclei_filtered (>=%.0f%% 重叠): %d 个细胞"
        % (min_overlap_fraction * 100, len(keep)))
    return df_match, keep


def report(df):
    frac = df["overlap"] / df["size_pix_histology"]
    for th in (0.3, 0.4, 0.5, 0.6):
        k = df[frac >= th].sort_values(["id_xenium", "overlap"],
                                       ascending=[True, False]).drop_duplicates("id_xenium")
        log("   阈值 %.1f -> %d 个细胞" % (th, len(k)))


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Step 4 的 min_overlap_fraction，默认 0.5")
    ap.add_argument("--dry_run", action="store_true", help="只报告，不写文件")
    ap.add_argument("--no_auto_params", action="store_true",
                    help="跳过参数搜索，直接用已知最优 (dy=1, dx=1, sc=1)")
    a = ap.parse_args(argv)

    with tifffile.TiffFile(FP_HESEG) as t:
        he_shape = t.series[0].shape
    orig_cols, orig_rows = he_shape[0], he_shape[1]
    log("H&E 分割图 %s -> 原始 H&E 尺寸 (rows=%d, cols=%d)"
        % (he_shape, orig_rows, orig_cols))

    # Xenium 帧取 morphology_focus（Xenium 自己的成像画布）的物理范围，
    # 它是权威边界；用细胞坐标最大值会切掉边界上的核。
    with tifffile.TiffFile(FP_MORPH) as t:
        pgs = t.pages[0].shape          # (C, Y, X)
        mph, mpw = int(pgs[-2]), int(pgs[-1])
    rh = int(np.ceil(mph * PXM))
    rw = int(np.ceil(mpw * PXM))
    log("Xenium 帧 %d x %d um (morphology_focus %d x %d px @ %.4f um)"
        % (rw, rh, mpw, mph, PXM))

    cells = pd.read_csv(FP_CELLS, usecols=["cell_id", "x_centroid", "y_centroid"])

    xmap = pd.read_csv(FP_XMAP, index_col=0)["cell_id_num"]
    cells["num"] = cells["cell_id"].map(xmap)

    log("构建正确的 Xenium 核分割图 ...")
    xen = build_xenium_mask(rh, rw)
    xenb = xen > 0

    m = np.loadtxt(FP_ALIGN, delimiter=",")
    a3 = m.copy()
    a3[2] = [0, 0, 1]
    a3i = np.linalg.inv(a3)
    log("官方配准矩阵 尺度=%.5f  0.2125*尺度=%.5f (H&E 像素 0.27377)"
        % (np.hypot(m[0, 0], m[1, 0]), PXM * np.hypot(m[0, 0], m[1, 0])))

    # ---- 参数搜索在 he_image_nuclei_seg_microns.tif 上做（快，且已在盘上）----
    mic = tifffile.imread(FP_HEMICRON)
    log("he microns %s, 标签 %d" % (mic.shape, len(np.unique(mic)) - 1))

    def iou_of(dy, dx, sc):
        w = warp_he(mic, rh, rw, a3i, orig_rows, orig_cols, dy, dx, sc,
                    idx_scale=S_MICRON)
        w = w > 0
        return float((xenb & w).sum()) / float((xenb | w).sum())

    # 错位基准：把 he_microns 直接按相同索引放进 Xenium 帧（当前管线的事实行为）
    _naive = np.zeros((rh, rw), bool)
    _h, _w = min(rh, mic.shape[0]), min(rw, mic.shape[1])
    _naive[:_h, :_w] = mic[:_h, :_w] > 0
    log("错位基准（同索引对齐）IoU = %.4f"
        % (float((xenb & _naive).sum()) / float((xenb | _naive).sum())))

    if a.no_auto_params:
        dy, dx, sc = 1.0, 1.0, 1.0
        log("跳过搜索，用 dy=%.1f dx=%.1f sc=%.3f" % (dy, dx, sc))
    else:
        log("搜索最优偏移与尺度 ...")
        best = None
        for ddy in np.arange(-4.0, 4.1, 2.0):
            for ddx in np.arange(-4.0, 4.1, 2.0):
                v = iou_of(ddy, ddx, 1.0)
                if best is None or v > best[0]:
                    best = (v, float(ddy), float(ddx), 1.0)
        for ddy in np.arange(best[1] - 2, best[1] + 2.1, 1.0):
            for ddx in np.arange(best[2] - 2, best[2] + 2.1, 1.0):
                v = iou_of(ddy, ddx, 1.0)
                if v > best[0]:
                    best = (v, float(ddy), float(ddx), 1.0)
        for s in np.arange(0.98, 1.021, 0.01):
            v = iou_of(best[1], best[2], s)
            if v > best[0]:
                best = (v, best[1], best[2], float(s))
        _, dy, dx, sc = best
        log("最优参数 dy=%.1f dx=%.1f scale=%.3f (microns 域 IoU=%.4f)" % (dy, dx, sc, best[0]))

    # ---- 用同一组参数在 FULL-RES H&E 分割图上做一次最终重采样 ----
    log("从全分辨率 H&E 分割图重采样（保留标签，避免二次插值）...")
    he_seg = tifffile.imread(FP_HESEG)
    log("  read %s, 标签 %d" % (he_seg.shape, len(np.unique(he_seg)) - 1))
    hew = warp_he(he_seg, rh, rw, a3i, orig_rows, orig_cols, dy, dx, sc)
    del he_seg
    hb = hew > 0
    log("最终配准 IoU = %.4f" % (float((xenb & hb).sum()) / float((xenb | hb).sum())))
    log("  标签: Xenium %d, H&E %d" % (len(np.unique(xen)) - 1, len(np.unique(hew)) - 1))

    # ---- 配对正确性检查：配对的两个核中心应几乎重合 ----
    def centroids(mask, maxv):
        rows = np.repeat(np.arange(mask.shape[0], dtype=np.int64), mask.shape[1])
        cols = np.tile(np.arange(mask.shape[1], dtype=np.int64), mask.shape[0])
        v = mask.ravel()
        nz = v > 0
        lab = v[nz].astype(np.int64)
        n = np.bincount(lab, minlength=maxv)
        rs = np.bincount(lab, weights=rows[nz], minlength=maxv)
        cs = np.bincount(lab, weights=cols[nz], minlength=maxv)
        keep = n > 0
        return np.nonzero(keep)[0], rs[keep] / n[keep], cs[keep] / n[keep]

    df_match, keep = match(xen, hew, a.threshold)
    report(df_match)

    h_ids, h_rs, h_cs = centroids(hew, int(hew.max()) + 1)
    x_ids, x_rs, x_cs = centroids(xen, int(xen.max()) + 1)
    hmap = {int(k): i for i, k in enumerate(h_ids)}
    xmap2 = {int(k): i for i, k in enumerate(x_ids)}
    d = []
    for h, x in zip(keep["id_histology"].to_numpy(), keep["id_xenium"].to_numpy()):
        i, j = hmap.get(int(h)), xmap2.get(int(x))
        if i is not None and j is not None:
            d.append(np.hypot(h_rs[i] - x_rs[j], h_cs[i] - x_cs[j]))
    d = np.array(d)
    log("配对中心距 (um): 中位 %.2f  均值 %.2f  p90 %.2f  <3um 占比 %.3f  (n=%d)"
        % (np.median(d), d.mean(), np.percentile(d, 90), float((d < 3).mean()), len(d)))

    if a.dry_run:
        log("--dry_run：不写任何文件")
        return 0

    # ---- 备份 + 写出 ----
    bk = "output/_backup_before_fix"
    os.makedirs(bk, exist_ok=True)
    for fp in ["output/matched_nuclei.csv", "output/matched_nuclei_filtered.csv",
               "output/cell_gene_matrix_filtered.csv"]:
        if os.path.exists(fp):
            dst = os.path.join(bk, os.path.basename(fp))
            if not os.path.exists(dst):
                shutil.copy2(fp, dst)
                log("备份 %s -> %s" % (fp, dst))

    df_match.to_csv("output/matched_nuclei.csv", index=False)
    keep.to_csv("output/matched_nuclei_filtered.csv", index=False)

    mapping = dict(zip(keep["id_xenium"].astype(int), keep["id_histology"].astype(int)))
    expr = pd.read_csv(FP_CGM, index_col=0)
    idx = pd.Index([mapping.get(int(i), -1) for i in expr.index])
    expr = expr.assign(_h=idx)
    expr = expr[expr["_h"] > 0].set_index("_h")
    expr.index.name = None
    expr = expr.loc[sorted(expr.index)]
    expr.to_csv("output/cell_gene_matrix_filtered.csv")
    log("cell_gene_matrix_filtered.csv: %d 细胞 x %d 基因" % expr.shape)

    tifffile.imwrite("output/xenium_nuclei_fixed.tif", xen, photometric="minisblack")

    dd = 4
    ov = np.zeros((rh // dd, rw // dd, 3), np.uint8)
    sx = xenb[: rh // dd * dd, : rw // dd * dd].reshape(rh // dd, dd, rw // dd, dd).any((1, 3))
    sh = hb[: rh // dd * dd, : rw // dd * dd].reshape(rh // dd, dd, rw // dd, dd).any((1, 3))
    ov[..., 0] = np.where(sh, 255, 0)
    ov[..., 1] = np.where(sx, 255, 0)
    Image.fromarray(ov).save("output/registration_overlay.png")
    log("已写出 output/registration_overlay.png（红=H&E 核, 绿=Xenium 核, 黄=重叠）")
    log("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
