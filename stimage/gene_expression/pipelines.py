# pipelines.py
import mlflow
from .preprocessing import *


def run_pipeline(adata, pipeline):
    """
    Run the preprocessing pipeline and log each step to MLflow.
    """
    for step_name, params in pipeline['steps']:
        # Log the step being executed
        mlflow.log_param(f"{step_name}_params", params)

        # Execute the preprocessing step
        adata = globals()[step_name](adata, **params)

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
    }
]
# yapf: enable
