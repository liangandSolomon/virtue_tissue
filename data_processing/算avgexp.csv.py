import pandas as pd
expr = pd.read_csv("cell_gene_matrix_filtered.csv", index_col=0)
ct   = pd.read_csv("celltype_filtered.csv", index_col="c_id")
common = expr.index.intersection(ct.index)
avg = expr.loc[common].join(ct.loc[common,"ct"]).groupby("ct").mean()
avg = avg.loc[["B","Myeloid","Endothelial","Fibroblast","Macrophage",
                 "Malignant","Epithelial","Plasma","T"]]
avg.to_csv("avgexp.csv")