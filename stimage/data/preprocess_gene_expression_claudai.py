import polars as pl
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path


# Function to read and process a single CSV file
def read_csv_file(file_path, image_id):
    df = pl.read_csv(file_path)
    df = df.with_columns(pl.lit(image_id).alias('image_id'))
    return df


# Directory containing your CSV files
data_dir = Path('path/to/your/data/directory')

# List all CSV files in the directory
csv_files = list(data_dir.glob('*.csv'))

# Read and concatenate all CSV files
dfs = []
for i, file in enumerate(csv_files):
    df = read_csv_file(file, f'image_{i+1}')
    dfs.append(df)

combined_df = pl.concat(dfs)

# Convert the combined Polars DataFrame to a pandas DataFrame
# (necessary for creating AnnData object)
pd_df = combined_df.to_pandas()

# Create AnnData object
adata = sc.AnnData(X=pd_df.drop(columns=['image_id']),
                   obs=pd_df[['image_id']],
                   var=pd.DataFrame(index=pd_df.columns.drop('image_id')))

# Basic preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Normalize the data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='image_id')

# Scale the data
sc.pp.scale(adata, max_value=10)

# Perform batch correction
sc.pp.combat(adata, key='image_id')

# Perform PCA on the batch-corrected data
sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

print(f"Shape of PCA result: {adata.obsm['X_pca'].shape}")
print(adata)

# Optional: Save the preprocessed data
adata.write('preprocessed_spatial_data.h5ad')

# Visualization (optional)
sc.pl.pca(adata, color='image_id')
sc.pl.umap(adata, color='image_id')
