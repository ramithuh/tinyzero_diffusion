# tzd: Minimal Hydra + PyTorch Lightning Project

This repository is a minimal, well-structured project demonstrating the integration of Hydra and PyTorch Lightning. It follows modern Python packaging best practices using a `src` layout.

## Project Structure

```
.
├── .gitignore
├── README.md
├── configs
│   └── config.yaml
├── notebooks
│   └── exploration.ipynb
├── pyproject.toml
├── scripts
│   └── train.py
├── src
│   └── tzd
│       ├── __init__.py
│       └── models.py
└── tests
    └── test_models.py
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd tzd
    ```

2.  **Create a conda environment with Python 3.11:**
    ```bash
    conda create -n tzd python=3.11
    conda activate tzd
    ```

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install the project in editable mode:**
    This command will install the project and its dependencies listed in `pyproject.toml`. The `-e` flag allows you to make changes to the source code and have them immediately reflected without reinstalling.
    ```bash
    pip install -e .
    ```

## Usage

### Running the Training Script

To run the training script, use the following command from the project root:

```bash
python scripts/train.py
```

### Overriding Configuration

Hydra allows you to easily override any configuration parameter from the command line.

For example, to change the number of training epochs or the batch size:

```bash
python scripts/train.py trainer.max_epochs=20 data.batch_size=64
```

### Running Tests

To run the tests, you can use `pytest`:

```bash
pytest
```
