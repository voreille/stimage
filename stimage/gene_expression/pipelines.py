# pipelines.py
import logging
import mlflow
from .preprocessing import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def run_pipeline(adata, pipeline):
    """
    Run the preprocessing pipeline and log each step to both Python logging and MLflow.
    """
    logger.info(f"Starting pipeline: {pipeline['name']}")
    mlflow.log_param("pipeline_name",
                     pipeline['name'])  # Log the name of the pipeline

    for step_name, params in pipeline['steps']:
        # Log the step being executed to both MLflow and the logger
        logger.info(f"Executing step: {step_name} with parameters: {params}")
        mlflow.log_param(f"{step_name}_params", params)

        # Execute the preprocessing step
        try:
            adata = globals()[step_name](adata, **params)
            logger.info(f"Step {step_name} completed successfully.")
        except Exception as e:
            logger.error(f"Error during step {step_name}: {str(e)}")
            mlflow.log_metric(f"{step_name}_failed", 1)
            raise e  # Re-raise the exception to stop the pipeline

    logger.info(f"Pipeline {pipeline['name']} completed successfully.")
    return adata


# yapf: disable
pipelines = [
    {
        'name': 'pipeline_1',
        'steps': [
            ('filter_cells', {'min_genes': 200}),
            ('filter_genes', {'min_cells': 3}),
            ('normalize_total', {'target_sum': 1e4}),
            ('log1p', {}),
            ('highly_variable_genes', {'n_top_genes': 2000, 'batch_key': 'image_id'}),
            ('scale', {'max_value': 10}),
            ('combat', {'key': 'image_id'}),
            ('pca', {'n_comps': 128, 'svd_solver': 'arpack'})
        ]
    },
    {
        'name': 'pipeline_2',
        'steps': [
            ('filter_cells', {'min_genes': 150}),
            ('filter_genes', {'min_cells': 5}),
            ('normalize_total', {'target_sum': 1e5}),
            ('log1p', {}),
            ('highly_variable_genes', {'n_top_genes': 3000, 'batch_key': 'image_id'}),
            ('scale', {'max_value': 5}),
            ('combat', {'key': 'image_id'}),
            ('pca', {'n_comps': 128, 'svd_solver': 'randomized'})
        ]
    },
    {
        'name': 'pipeline_3_sctransform_harmony',
        'steps': [
            ('filter_cells', {'min_genes': 200}),
            ('filter_genes', {'min_cells': 3}),
            ('sctransform', {}),  # SCTransform for normalization
            ('highly_variable_genes', {'n_top_genes': 2000, 'batch_key': 'image_id'}),  # Highly variable genes
            ('pca', {'n_comps': 50, 'svd_solver': 'arpack'}),  # PCA on the HVGs
            ('harmony_batch_correction', {'batch_key': 'image_id'}),  # Apply Harmony
        ]
    },
    {
        'name': 'pipeline_4_scanorama',
        'steps': [
            ('filter_cells', {'min_genes': 150}),
            ('filter_genes', {'min_cells': 5}),
            ('normalize_total', {'target_sum': 1e4}),
            ('log1p', {}),
            ('highly_variable_genes', {'n_top_genes': 3000, 'batch_key': 'image_id'}),
            ('scanorama_integration', {'batch_key': 'image_id'}),
        ]
    }
]
# yapf: enable
