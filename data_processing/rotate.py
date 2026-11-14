import tifffile, numpy as np

fp_in  = "Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif"
fp_out = "Xenium_V1_humanLung_Cancer_FFPE_he_image_rot90ccw.ome.tif"

with tifffile.TiffFile(fp_in) as tf:
    he = tf.series[0].asarray()      # 显存不够可分块/降分辨率处理
he_rot = np.rot90(he, k=1)           # 逆时针 90°
tifffile.imwrite(fp_out, he_rot, photometric='rgb')