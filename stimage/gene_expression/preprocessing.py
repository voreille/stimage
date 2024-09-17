import scanpy as sc
import scvelo as scv
import scanorama
import harmonypy as hm
import pandas as pd

# Global registry for steps
STEP_REGISTRY = {}


def register_step(func):
    """Decorator to register a preprocessing function in the registry."""
    STEP_REGISTRY[func.__name__] = func
    return func


@register_step
def filter_cells(adata, min_genes):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    return adata


@register_step
def filter_genes(adata, min_cells):
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


@register_step
def normalize_total(adata, target_sum):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    adata.uns['normalize_total'] = {
        'target_sum': target_sum
    }  # Store for reversal
    return adata


@register_step
def log1p(adata):
    sc.pp.log1p(adata)
    return adata


@register_step
def highly_variable_genes(adata, n_top_genes, batch_key):
    sc.pp.highly_variable_genes(adata,
                                n_top_genes=n_top_genes,
                                batch_key=batch_key)
    return adata


@register_step
def scale(adata, max_value):
    sc.pp.scale(adata, max_value=max_value)
    adata.uns['scale'] = {'max_value': max_value}  # Store for reversal
    return adata


@register_step
def combat(adata, key):
    sc.pp.combat(adata, key=key)
    adata.uns['combat'] = {'key': key}  # Store for reversal
    return adata


@register_step
def pca(adata, n_comps, svd_solver):
    sc.tl.pca(adata, n_comps=n_comps, svd_solver=svd_solver)
    return adata


@register_step
def sctransform(adata):
    """
    SCTransform for normalization.
    """
    scv.pp.filter_and_normalize(adata, log=True)
    return adata


@register_step
def harmony_batch_correction(adata, batch_key):
    """
    Apply Harmony batch correction on PCA results.
    """
    # Ensure that PCA has been run
    if 'X_pca' not in adata.obsm:
        raise ValueError('PCA needs to be computed before running Harmony.')

    # Extract PCA results
    pca_data = adata.obsm['X_pca']

    # Extract metadata (batch information)
    meta_data = pd.DataFrame(adata.obs[batch_key])

    # Run Harmony batch correction
    ho = hm.run_harmony(pca_data, meta_data, [batch_key])

    # Store the corrected PCA result back into the AnnData object
    adata.obsm['X_pca_harmony'] = ho.Z_corr.T

    return adata


@register_step
def scanorama_integration(adata_list, batch_key):
    """
    Scanorama for batch correction and integration.
    """
    # Split adata based on the batch_key
    adatas_split = [
        adata[adata.obs[batch_key] == batch].copy()
        for batch in adata.obs[batch_key].unique()
    ]

    # Perform Scanorama integration
    integrated_data, corrected = scanorama.integrate_scanpy(adatas_split,
                                                            dimred=50)

    # Merge back the corrected datasets
    adata = adatas_split[0].concatenate(adatas_split[1:])

    return adata


@register_step
def umap(adata):
    """
    UMAP for dimensionality reduction.
    """
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
    sc.tl.umap(adata)
    return adata
