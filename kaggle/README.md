# Run pipeline accuracy on Kaggle (GPU)

This folder is a **Kaggle Script** kernel that clones the repo, attaches the MUSAE Facebook dataset, runs `tests/testing_pipeline_accuracy.py` with **heavy mode** (wider graph-RAG search + optional GNN fine-tune), and writes metrics under `/kaggle/working`.

## Prerequisites

- [Kaggle API](https://github.com/Kaggle/kaggle-api) installed and `~/.kaggle/kaggle.json` valid (`kaggle kernels list` works).
- From the repo root, push with `python scripts/push_kaggle_kernel.py` (loads `KAGGLE_USERNAME` / `KAGGLE_KEY` from `.env`).

## Push and run

```bash
python scripts/push_kaggle_kernel.py
```

Or from this directory (after fixing `kernel-metadata.json` `id` to `your_username/graphrag-pipeline-accuracy`):

```bash
kaggle kernels push -p path/to/repo/kaggle
```

## Status and outputs

```bash
kaggle kernels status YOUR_USERNAME/graphrag-pipeline-accuracy
kaggle kernels output YOUR_USERNAME/graphrag-pipeline-accuracy -p ./kaggle_output
```

Artifacts (when the test completes):

- `pipeline_accuracy_results.json` — accuracies, best hyperparameters, wall time.
- `kaggle_kernel_exit_code.txt` — pytest exit code.

## Environment variables (optional)

| Variable | Effect |
|----------|--------|
| `PIPELINE_ACCURACY_HEAVY=1` | Large grid + GNN fine-tune (default on Kaggle GPU via `run_pipeline_test.py`). |
| `PIPELINE_ACCURACY_FAST=1` | Small grid, no fine-tune (for quick CPU checks). |
| `PIPELINE_FINETUNE_EPOCHS` | Max epochs for fine-tune (default `80`). |
| `PIPELINE_FINETUNE_LR` | AdamW LR (default `5e-4`). |
| `GRAPH_RAG_REPO_URL` | Git URL if not using the default GitHub remote. |

## Kernel metadata

- **GPU** and **internet** are enabled in `kernel-metadata.json`.
- **Dataset**: `rozemberczki/musae-facebook-pagepage-network`.

## Re-run after edits

Push again from the repo (or use `kaggle kernels push -p kaggle` with your metadata). For faster iteration on Kaggle, see [Kaggle kernels documentation](https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata) for draft runs.
