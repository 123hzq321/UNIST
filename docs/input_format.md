# UNIST Input Format

UNIST uses a directory-based format so datasets can be created from Visium, Xenium, CosMx, pseudo-spots or other spatial assays without changing the model code.

## Reference-supervised real spatial query

Required files:

```text
dataset/
  spatial.h5ad
  sc_reference.h5ad
  metadata.json
```

`spatial.h5ad`

- Rows are spatial units: spots, bins, cells or grid spots.
- Columns are genes.
- `obs_names` are used as `spot_id` values in outputs.
- `var_names` must be gene names shared with the reference.

`sc_reference.h5ad`

- Rows are single cells.
- Columns are genes in the same naming scheme as `spatial.h5ad`.
- `obs["cell_type"]` is used as the label column by default.

`metadata.json`

```json
{
  "dataset_id": "my_dataset",
  "sc_label_col": "cell_type"
}
```

`sc_label_col` is optional when the reference already contains `obs["cell_type"]`.

## Synthetic or gold pseudo-spot task

Required files:

```text
dataset/
  spatial.h5ad
  metadata.json
  ground_truth_proportions.csv
```

`metadata.json` must include:

```json
{
  "dataset_id": "my_gold_task",
  "known_cell_types": ["Tumor", "T cell", "CAF"]
}
```

`ground_truth_proportions.csv` must include:

```text
spot_id,cell_type,true_proportion
spot_1,Tumor,0.7
spot_1,T cell,0.2
spot_1,CAF,0.1
```

## Outputs

All UNIST runs write the same output schema:

- `predicted_labels.csv`
- `uncertainty.csv`
- `spot_by_celltype.csv`
- `metrics.csv`
- `run_metadata.json`
- `training_losses.json`

