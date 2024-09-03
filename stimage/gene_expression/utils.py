from pathlib import Path
import tempfile

import polars as pl
import mlflow
import pandas as pd
from tqdm import tqdm
import click
import scanpy as sc


def read_csv_file(file_path, image_id, gene_name_overlap):
    df = pl.read_csv(file_path)
    df = df.select([col for col in df.columns if col in gene_name_overlap])
    df = df.with_columns(pl.lit(image_id).alias('image_id'))
    return df


def compute_gene_name_overlap(data, data_dir):
    gene_name_overlap = None
    for i, ind in enumerate(data.index):
        gene_exp_slide = pd.read_csv(
            f'{data_dir}/{data.tech[ind]}/gene_exp/{data.slide[ind]}_count.csv',
            sep=',',
            nrows=1,
            index_col=0)
        if i == 0:
            gene_name_overlap = set(gene_exp_slide.columns)
        else:
            gene_name_overlap = gene_name_overlap.intersection(
                gene_exp_slide.columns)
        print(
            f"Processed file {i + 1}/{len(data.index)} for gene name overlap.")
    return gene_name_overlap


def load_and_filter_metadata(meta_file, species, tissue):
    meta = pd.read_csv(meta_file)
    data = meta.loc[(meta['species'] == species) &
                    (meta['tissue'] == tissue), :]
    return data


def read_and_combine_files(data, data_dir, gene_name_overlap):
    dfs = []
    for index in tqdm(range(len(data)), desc="Reading slides"):
        slide = data['slide'].iloc[index]
        tech = data['tech'].iloc[index]

        # Construct the file path
        file_path = Path(f'{data_dir}/{tech}/gene_exp/{slide}_count.csv')

        if file_path.exists():
            df = read_csv_file(file_path, slide, gene_name_overlap)
            dfs.append(df)
        else:
            click.echo(
                f"Warning: File {file_path} does not exist and will be skipped."
            )

    if not dfs:
        click.echo("No valid CSV files found. Exiting.")
        return None

    combined_df = pl.concat(dfs)
    return combined_df


def plot_and_save_figures(adata, save_figures_dir, pca_components):
    save_figures_dir = Path(save_figures_dir).resolve()
    save_figures_dir.mkdir(exist_ok=True)

    # Visualization: PCA colored by image_id (batch effect)
    pca_plot = sc.pl.pca(adata,
                         color='image_id',
                         title='PCA colored by image_id (batch effect)',
                         show=False)

    # Perform t-SNE on the batch-corrected data
    sc.tl.tsne(adata, n_pcs=pca_components)

    # Visualization: t-SNE colored by image_id
    tsne_plot = sc.pl.tsne(adata,
                           color='image_id',
                           title='t-SNE colored by image_id',
                           show=False)

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)

        # Save the PCA plot
        pca_file = temp_dir / 'pca_plot.png'
        pca_plot.figure.savefig(pca_file, bbox_inches='tight')

        # Log the PCA figure as an artifact in MLflow
        mlflow.log_artifact(pca_file, artifact_path="figures")

        # Save the t-SNE plot
        tsne_file = temp_dir / 'tsne_plot.png'
        tsne_plot.figure.savefig(tsne_file, bbox_inches='tight')

        # Log the t-SNE figure as an artifact in MLflow
        mlflow.log_artifact(tsne_file, artifact_path="figures")

        click.echo(f"Figures saved and logged to MLflow.")
