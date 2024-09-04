import os
from pathlib import Path
import json

import click
import mlflow
import scanpy as sc
import pandas as pd

from stimage.gene_expression.pipelines import pipelines, run_pipeline
from stimage.gene_expression.utils import (compute_gene_name_overlap,
                                           load_and_filter_metadata,
                                           read_and_combine_files,
                                           plot_and_save_figures)

project_dir = Path(__file__).resolve().parents[2]
os.environ['MLFLOW_TRACKING_URI'] = f"file://{project_dir / 'mlruns'}"
pipelines_choice = [p['name'] for p in pipelines]


@click.command()
@click.option('--data_dir',
              default=str((project_dir / 'data' / 'raw').resolve()),
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Directory containing your data organized by technology.')
@click.option('--species',
              type=click.STRING,
              default="human",
              help='Species to filter by in the meta file.')
@click.option('--tissue',
              type=click.STRING,
              default="lung",
              help='Tissue to filter by in the meta file.')
@click.option('--output_dir',
              type=click.Path(file_okay=False, dir_okay=True),
              default=str(project_dir / "data/processed/gene_expression/"),
              help='Output file for the preprocessed data.')
@click.option('--meta_file',
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default=str((project_dir / 'data/meta_all_gene.csv').resolve()),
              help='Path to the metadata CSV file.')
@click.option(
    '--save_figures',
    type=click.Path(file_okay=False, dir_okay=True),
    # default=str((project_dir / "reports/gene_exp_preprocessing").resolve()),
    default=None,
    help=
    'Directory to save the figures. If not specified, figures will be displayed.'
)
@click.option('--pipeline',
              type=click.Choice(pipelines_choice, case_sensitive=False),
              default="pipeline_3_sctransform_harmony",
              required=True,
              help='Specify which pipeline to use for preprocessing.')
@click.option('--obs_limit_per_file',
              type=click.INT,
              default=10000,
              help='Specify the limit of rows it will take from the csv')
def process_spatial_data(
    data_dir,
    species,
    tissue,
    output_dir,
    meta_file,
    save_figures,
    pipeline,
    obs_limit_per_file,
):

    # Load and filter metadata
    data = load_and_filter_metadata(meta_file, species, tissue)

    # Compute overlapping gene names across all slides
    gene_name_overlap = compute_gene_name_overlap(data, data_dir)
    if not gene_name_overlap:
        click.echo("No overlapping gene names found. Exiting.")
        return

    # Read and combine all relevant CSV files
    combined_df = read_and_combine_files(
        data,
        data_dir,
        gene_name_overlap,
        row_limit=obs_limit_per_file,
    )
    if combined_df is None:
        return

    # Create AnnData object
    pd_df = combined_df.to_pandas()
    adata = sc.AnnData(
        X=pd_df.drop(columns=['image_id', 'spots_id']),
        obs=pd_df[['image_id', 'spots_id']],
        var=pd.DataFrame(index=pd_df.drop(
            columns=["image_id", "spots_id"]).columns),
    )

    # Save raw counts before preprocessing
    # adata.raw = adata

    selected_pipeline = next(p for p in pipelines if p['name'] == pipeline)
    with mlflow.start_run(
            run_name=f"gene_exp_{selected_pipeline['name']}_debug"):
        # Log the pipeline name
        mlflow.set_tag("pipeline_name",
                       f"gene_exp_{selected_pipeline['name']}")

        # Run the preprocessing pipeline
        adata = run_pipeline(adata, selected_pipeline)

        # Save the processed data as an artifact
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{selected_pipeline['name']}_{species}_{tissue}_debug.h5ad"
            adata.write(output_file)
        mlflow.log_artifact(output_file)

        # Save figures if save_figures option is specified
        if save_figures:
            pca_components = next(step[1]['n_comps']
                                  for step in selected_pipeline['steps']
                                  if step[0] == 'pca')
            plot_and_save_figures(adata, save_figures, pca_components)

        print(f"Completed and logged {selected_pipeline['name']}")


if __name__ == '__main__':
    process_spatial_data()
