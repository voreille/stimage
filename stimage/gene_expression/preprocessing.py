import scanpy as sc


def filter_cells(adata, min_genes):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    return adata


def filter_genes(adata, min_cells):
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


def normalize_total(adata, target_sum):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    adata.uns['normalize_total'] = {
        'target_sum': target_sum
    }  # Store for reversal
    return adata


def log1p(adata):
    sc.pp.log1p(adata)
    return adata


def highly_variable_genes(adata, n_top_genes, batch_key):
    sc.pp.highly_variable_genes(adata,
                                n_top_genes=n_top_genes,
                                batch_key=batch_key)
    return adata


def scale(adata, max_value):
    sc.pp.scale(adata, max_value=max_value)
    adata.uns['scale'] = {'max_value': max_value}  # Store for reversal
    return adata


def combat(adata, key):
    sc.pp.combat(adata, key=key)
    adata.uns['combat'] = {'key': key}  # Store for reversal
    return adata


def pca(adata, n_comps, svd_solver):
    sc.tl.pca(adata, n_comps=n_comps, svd_solver=svd_solver)
    return adata
