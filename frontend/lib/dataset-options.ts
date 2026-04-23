/**
 * Shared dataset / GNN options for the UI (aligned with api/schemas VALID_DATASETS + GNN weights).
 */

export const DATASET_SCOPE_OPTIONS = [
  { value: "all", label: "All Datasets" },
  { value: "facebook", label: "Facebook" },
  { value: "twitter", label: "Twitter" },
  { value: "reddit", label: "Reddit" },
  { value: "demo", label: "Demo" },
] as const;

/** Which pretrained checkpoint to run. Empty = follow dataset row (all/demo → Facebook GNN). */
export const GNN_MODEL_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "GNN: Auto (from dataset, else Facebook)" },
  { value: "facebook", label: "GNN: Facebook weights" },
  { value: "twitter", label: "GNN: Twitter weights" },
  { value: "reddit", label: "GNN: Reddit weights" },
];
