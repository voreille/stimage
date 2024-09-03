from pathlib import Path
import logging

import pandas as pd
from PIL import Image
import click
from tqdm import tqdm

# Set the maximum image pixels limit
Image.MAX_IMAGE_PIXELS = 3080000000000000000000000000000000000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crop_image_all(path, slide, species, tissue, parent_dir):
    # Open the image and convert to RGB
    logging.info(f"Processing slide: {slide} for species: {species}, tissue: {tissue}")
    temp_image = Image.open(path / f'image/{slide}.png')
    temp_image = temp_image.convert('RGB')

    # Read the coordinates from CSV
    coord = pd.read_csv(path / f'coord/{slide}_coord.csv')
    r = coord.r[0]

    # Create the output directory if it doesn't exist
    output_dir = parent_dir / f'data/processed/patches/{species}_{tissue}/'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop through the coordinates and crop the image
    for i in tqdm(range(len(coord.index)), desc=f"Processing {slide}", unit="patch"):
        yaxis = coord.yaxis[i]
        xaxis = coord.xaxis[i]
        spot_name = coord.iloc[i, 0]
        temp_image_crop = temp_image.crop(
            (xaxis - r, yaxis - r, xaxis + r, yaxis + r))
        temp_image_crop.save(output_dir / f"{spot_name}.png")


@click.command()
@click.option('--species',
              type=click.Choice(['human', 'mouse']),
              default='human',
              help='Species to filter the meta dataframe')
@click.option('--tissue',
              type=click.Choice(['lung', 'brain']),
              default='lung',
              help='Tissue type to filter the meta dataframe')
def main(species, tissue):
    # Determine the parent directory (two levels up)
    script_path = Path(__file__).resolve()
    parent_dir = script_path.parents[2]

    # Read the meta dataframe from the data folder within the parent directory
    meta_file = parent_dir / 'data/meta_all_gene.csv'
    meta = pd.read_csv(meta_file)

    # Apply filters if provided
    if species:
        meta = meta[meta['species'] == species]
    if tissue:
        meta = meta[meta['tissue'] == tissue]

    # Iterate over all rows in the filtered meta dataframe
    for index, row in tqdm(meta.iterrows()):
        slide = row['slide']
        tech = row['tech']

        # Generate the path relative to the parent directory and call crop_image_all
        path = parent_dir / f"data/raw/{tech}"
        crop_image_all(path, slide, species, tissue, parent_dir)


if __name__ == '__main__':
    main()
