
'''把 10x 注释表转成 GHIST 需要的 celltype_filtered.csv（c_id = id_histology）。'''
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

name_map = {  # 10x 注释的缩写 -> config_lung.json 的 cell_types
    'epi': 'Malignant', 'normal_epi': 'Epithelial', 'fib': 'Fibroblast',
    'nkt': 'T', 'mph': 'Macrophage', 'endo': 'Endothelial',
    'plasma': 'Plasma', 'b': 'B', 'mast': 'Myeloid', 'cil': 'Epithelial',
}


def log(*a):
    print(*a, flush=True)


def main():
    ann = pd.read_csv('xenium_luad_by_annotation.csv')              # cell_id,group,color
    xmap = pd.read_csv('xenium_cell_ids_dict.csv', index_col=0)     # cell_id -> cell_id_num
    matched = pd.read_csv('output/matched_nuclei_filtered.csv')     # id_xenium -> id_histology
    expr_ids = pd.read_csv('output/cell_gene_matrix_filtered.csv',
                           index_col=0, usecols=[0]).index           # index = id_histology

    n_ann = len(ann)

    # 1) 10x 字符串 ID -> Xenium 数字 ID
    ann['id_xenium'] = ann['cell_id'].map(xmap['cell_id_num'])
    n_xe = int(ann['id_xenium'].notna().sum())
    out = ann.dropna(subset=['id_xenium']).copy()
    out['id_xenium'] = out['id_xenium'].astype(int)
    log('annotation rows %d -> id_xenium %d (-%d)' % (n_ann, n_xe, n_ann - n_xe))

    # 2) id_xenium -> id_histology（关键的一步，旧版缺失）
    out = out.merge(matched[['id_xenium', 'id_histology']], on='id_xenium', how='inner')
    log('-> id_histology %d (-%d)' % (len(out), n_xe - len(out)))

    # 3) 只保留真正进入表达矩阵的细胞
    out = out[out['id_histology'].isin(set(expr_ids))].copy()
    log('-> in expression matrix %d' % len(out))

    out['c_id'] = out['id_histology'].astype(int)
    out['ct'] = out['group'].map(name_map)

    # 同一个 id_histology 可能被多个注释命中；保留第一条，避免训练侧索引歧义
    dup = int(out['c_id'].duplicated().sum())
    if dup:
        out = out.drop_duplicates(subset='c_id', keep='first')
        log('dropped %d duplicate id_histology' % dup)

    unmapped = out.loc[out['ct'].isna(), 'group'].unique().tolist()
    assert not unmapped, 'unmapped annotation groups: %s' % unmapped
    assert out['c_id'].is_unique
    assert out['id_histology'].is_unique

    out[['c_id', 'ct']].to_csv('output/celltype_filtered.csv', index=False)
    log('saved output/celltype_filtered.csv: %d cells / %d classes'
        % (len(out), out['ct'].nunique()))
    log(out['ct'].value_counts().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
