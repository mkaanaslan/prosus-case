# Product Search Demo

## Overview

This repository contains code for running a recommendation/embedding-based experiment. The main entry point is `main.py`, which loads pre-computed embeddings and runs the full pipeline. If you wish to rebuild embeddings or rerun baseline and improved models from scratch, follow the instructions under [Re-running Experiments](#re-running-experiments).

## Setup

1. **Clone the repository**:

   ```bash
   git clone <REPO_URL>
   cd <REPO_FOLDER>
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Before running any script, make sure you have the dataset file:

* Download or copy `5k_items_curated.csv` into the `data/` folder at the root of the repository:

  ```bash
  mkdir -p data
  cp /path/to/5k_items_curated.csv data/
  ```

## Running the Application

The `main.py` script uses pre-generated embeddings and runs the complete pipeline end-to-end.

```bash
python main.py
```

Results will be saved in the default output directory, and logs will be printed to the console.

## Re-running Experiments

If you need to rebuild embeddings and rerun experiments from scratch:

1. **Enable update mode**:

   * Open `baseline.py` and set:

     ```python
     UPDATE = True
     ```
2. **Run baseline experiment**:

   ```bash
   python baseline.py
   ```
3. **Run improved experiment**:

   ```bash
   python improved.py
   ```

This will generate fresh embeddings, save them to disk, and rerun the baseline and improved models.

## Requirements

All Python dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```
