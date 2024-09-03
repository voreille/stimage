from pathlib import Path

import click
import polars as pl
import scanpy as sc
import pandas as pd
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[2]


# Function to read and process a single CSV file
def read_csv_file(file_path, image_id, gene_name_overlap):
    df = pl.read_csv(file_path)
    # Filter columns based on gene_name_overlap
    df = df.select([col for col in df.columns if col in gene_name_overlap])
    df = df.with_columns(pl.lit(image_id).alias('image_id'))
    return df


# Function to compute gene name overlap across all slides
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


# Function to load and filter metadata
def load_and_filter_metadata(meta_file, species, tissue):
    meta = pd.read_csv(meta_file)
    data = meta.loc[(meta['species'] == species) &
                    (meta['tissue'] == tissue), :]
    return data


# Function to read and combine all relevant CSV files
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


# Function to preprocess data and perform PCA
def preprocess_and_run_pca(combined_df, n_top_genes, pca_components):
    # Convert to pandas DataFrame
    pd_df = combined_df.to_pandas()

    # Create AnnData object
    adata = sc.AnnData(X=pd_df.drop(columns=['image_id']),
                       obs=pd_df[['image_id']],
                       var=pd.DataFrame(index=pd_df.columns.drop('image_id')))

    # Save raw counts before normalization and scaling
    adata.raw = adata

    # Basic preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalize the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata,
                                n_top_genes=n_top_genes,
                                batch_key='image_id')

    # Scale the data
    sc.pp.scale(adata, max_value=10)

    # Perform batch correction
    sc.pp.combat(adata, key='image_id')

    # Perform PCA
    sc.tl.pca(adata, svd_solver='arpack', n_comps=pca_components)

    return adata


# Function to save figures
def plot_and_save_figures(adata, save_figures_dir, pca_components):
    save_figures_dir = Path(save_figures_dir).resolve()
    save_figures_dir.mkdir(exist_ok=True)

    # Visualization: PCA colored by image_id (batch effect)
    pca_plot = sc.pl.pca(adata,
                         color='image_id',
                         title='PCA colored by image_id (batch effect)',
                         show=False)
    pca_plot.figure.savefig(save_figures_dir / 'pca_plot.png',
                            bbox_inches='tight')

    # Perform t-SNE on the batch-corrected data
    sc.tl.tsne(adata, n_pcs=pca_components)

    # Visualization: t-SNE colored by image_id
    tsne_plot = sc.pl.tsne(adata,
                           color='image_id',
                           title='t-SNE colored by image_id',
                           show=False)
    tsne_plot.figure.savefig(save_figures_dir / 'tsne_plot.png',
                             bbox_inches='tight')

    click.echo(f"Figures saved in {save_figures_dir}.")


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
@click.option('--output',
              type=click.Path(file_okay=True, dir_okay=False),
              default=str(project_dir /
                          "data/processed/gene_expression/human_lung.hd5a"),
              help='Output file for the preprocessed data.')
@click.option('--n_top_genes',
              type=click.INT,
              default=2000,
              help='Number of top variable genes to select.')
@click.option('--pca_components',
              type=click.INT,
              default=50,
              help='Number of PCA components.')
@click.option('--meta_file',
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default=str((project_dir / 'data/meta_all_gene.csv').resolve()),
              help='Path to the metadata CSV file.')
@click.option(
    '--save_figures',
    type=click.Path(file_okay=False, dir_okay=True),
    default=str((project_dir / "reports/gene_exp_preprocessing").resolve()),
    help=
    'Directory to save the figures. If not specified, figures will be displayed.'
)
def process_spatial_data(data_dir, species, tissue, output, n_top_genes,
                         pca_components, meta_file, save_figures):
    """
    A CLI tool to process spatial transcriptomics data based on species and tissue.
    
    DATA_DIR is the directory containing your data organized by technology.
    """
    # Load and filter metadata
    data = load_and_filter_metadata(meta_file, species, tissue)

    # Compute overlapping gene names across all slides
    gene_name_overlap = compute_gene_name_overlap(data, data_dir)
    if not gene_name_overlap:
        click.echo("No overlapping gene names found. Exiting.")
        return

    # Read and combine all relevant CSV files
    combined_df = read_and_combine_files(data, data_dir, gene_name_overlap)
    if combined_df is None:
        return

    # Preprocess data and run PCA
    adata = preprocess_and_run_pca(combined_df, n_top_genes, pca_components)

    # Print PCA results shape
    print(f"Shape of PCA result: {adata.obsm['X_pca'].shape}")
    print(adata)

    # Optional: Save the preprocessed data
    if output:
        output = Path(output)
        output.parent.mkdir(exist_ok=True)
        adata.write(output)

    # Save figures if save_figures option is specified
    if save_figures:
        plot_and_save_figures(adata, save_figures, pca_components)


if __name__ == '__main__':
    process_spatial_data()
