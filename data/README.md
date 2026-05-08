# Data

Raw datasets, downloaded medical images, and generated processed data are not committed to this repository.

This project uses public dermoscopic image datasets, including ISIC 2019-style data and HAM10000 / ISIC 2018-style data. Dataset acquisition and preprocessing are handled through the reusable pipeline in `src/data.py`.

The pipeline uses KaggleHub to download the datasets locally, resolves the dataset-specific directory structures, standardises labels, combines metadata, applies balancing, and generates a processed index file for training and evaluation.

Generated data files, downloaded image folders, and model-ready dataset indexes remain outside version control. This keeps the repository lightweight and supports responsible handling of medical image data.
