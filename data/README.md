# Data

Raw datasets and medical images are not committed to this repository.

This project uses public dermoscopic image datasets, including ISIC 2019-style data and HAM10000 / ISIC 2018-style data. Dataset acquisition and preprocessing are handled through the reusable pipeline in `src/data.py`.

The local data directory is organised into raw and processed files:

```text
data/
├── raw/
│   ├── images/
│   └── metadata.csv
└── processed/
    └── index_balanced_skin_lesions.csv
```

Keeping raw medical images out of version control helps avoid large repository files and supports responsible handling of medical image data.
