import sys
import os
import argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_processing'))

from tiff_io import imread_tiff

# Import from visualise_results
from visualise_results import (
    load_experiment_config,
    load_predicted_expr,
    load_cell_types,
    build_spatial_map,
    build_celltype_map,
    plot_gene_expression,
    plot_cell_types
)

# Import segmentation module using importlib (file starts with number)
seg_module = importlib.import_module('data_processing.3_segment_nuclei_he_image')
crop_patches = seg_module.crop_patches
run_hovernet = seg_module.run_hovernet
combine_crops = seg_module.combine_crops

def run_segmentation(he_image_path, output_dir, dir_hovernet="hover_net", gpu_id=0, crop_size=2000, del_intm=True):
    """Run HoverNet segmentation pipeline on a single HE image."""
    # Load and prepare image
    he_img = imread_tiff(he_image_path)
    if he_img.shape[-1] == 3:
        pass
    elif he_img.shape[0] == 3:
        he_img = np.moveaxis(he_img, 0, -1)
    else:
        raise ValueError("RGB channel not 0 or 2")
    height, width = he_img.shape[0], he_img.shape[1]

    crop_dir = os.path.join(output_dir, "he_image_nuclei_seg_crops")
    hovernet_output_dir = crop_dir + "_hovernet"
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(hovernet_output_dir, exist_ok=True)

    print("Step 1: Cropping patches...")
    crop_patches(he_img, crop_dir, crop_size, height, width)

    print("Step 2: Running HoverNet...")
    run_hovernet(dir_hovernet, gpu_id, crop_dir, hovernet_output_dir)

    print("Step 3: Combining crops...")
    combine_crops(height, width, output_dir, hovernet_output_dir)

    seg_path = os.path.join(output_dir, "he_image_nuclei_seg.tif")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation not produced at {seg_path}")

    if del_intm:
        import shutil
        shutil.rmtree(crop_dir)
        shutil.rmtree(hovernet_output_dir)
        print("Intermediate files deleted.")

    return seg_path

def main():
    parser = argparse.ArgumentParser(description="End-to-end inference and visualization from HE image")
    parser.add_argument("--he_image", required=True, help="Path to HE image (e.g., XeniumHE_LUAD_No14.tif)")
    parser.add_argument("--experiment", default="demo", help="Experiment name (default: demo)")
    parser.add_argument("--epoch", type=int, default=3, help="Epoch number (default: 3)")
    parser.add_argument("--gene", default="CD8A", help="Gene to visualize (default: CD8A)")
    parser.add_argument("--genes_file", default=None, help="File with list of genes")
    parser.add_argument("--output_dir", default=None, help="Output directory for visualizations")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for HoverNet (default: 0)")
    parser.add_argument("--crop_size", type=int, default=2000, help="Crop size for segmentation (default: 2000)")
    parser.add_argument("--dir_hovernet", default="hover_net", help="Directory of HoverNet code (default: hover_net)")
    parser.add_argument("--skip_segmentation", action="store_true", help="Skip segmentation, use existing seg file")
    parser.add_argument("--segmentation_path", default=None, help="Path to existing segmentation map if skipping")
    parser.add_argument("--no_he", action="store_true", help="Do not overlay H&E background")
    parser.add_argument("--roi", nargs=4, type=int, default=None, help="ROI: x1 y1 x2 y2")
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = f"experiments/{args.experiment}/visualisations"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determine segmentation path
    if args.skip_segmentation:
        if args.segmentation_path is None:
            seg_path = os.path.join("data_processing", "he_image_nuclei_seg.tif")
            if not os.path.exists(seg_path):
                seg_path = os.path.join(os.path.dirname(args.he_image), "he_image_nuclei_seg.tif")
        else:
            seg_path = args.segmentation_path
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        print(f"Using existing segmentation: {seg_path}")
    else:
        # Run segmentation
        seg_dir = "data_processing"
        seg_path = run_segmentation(args.he_image, seg_dir, args.dir_hovernet, args.gpu_id, args.crop_size, del_intm=True)
        print(f"Segmentation saved to: {seg_path}")

    # Load configuration and prediction data
    config = load_experiment_config(args.experiment)
    df_expr = load_predicted_expr(args.experiment, args.epoch, config)
    df_ct, cell_types_ordered = load_cell_types(config)

    # Load HE image (from the provided path, not from experiment)
    he_img = None if args.no_he else imread_tiff(args.he_image)
    if he_img is not None:
        if he_img.ndim == 5:
            he_img = he_img[0,0,...]
        elif he_img.ndim == 4:
            he_img = he_img[0,...]
        elif he_img.ndim == 2:
            he_img = np.stack([he_img]*3, axis=-1)

    # Load nuclei segmentation
    nuclei_seg = tifffile.imread(seg_path)

    # Determine genes
    if args.genes_file and os.path.exists(args.genes_file):
        with open(args.genes_file) as f:
            genes = [line.strip() for line in f if line.strip()]
    else:
        genes = [args.gene]

    # Visualize each gene
    for gene in genes:
        print(f"\n=== Visualizing gene: {gene} ===")
        expr_map = build_spatial_map(nuclei_seg, df_expr, gene)
        output_suffix = gene
        if args.roi:
            output_suffix += f"_roi_{args.roi[0]}_{args.roi[1]}_{args.roi[2]}_{args.roi[3]}"
        plot_gene_expression(
            he_img, expr_map, gene,
            os.path.join(output_dir, f"{output_suffix}.png"),
            roi=args.roi,
        )

    # Also cell types if available
    if df_ct is not None:
        print("\n=== Visualizing cell types ===")
        ct_map, ct_ordered = build_celltype_map(nuclei_seg, df_ct, cell_types_ordered)
        plot_cell_types(
            he_img, ct_map, ct_ordered,
            os.path.join(output_dir, "cell_types.png"),
            roi=args.roi,
        )

    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
