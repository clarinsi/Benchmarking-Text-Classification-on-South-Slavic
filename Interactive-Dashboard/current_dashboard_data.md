# Dashboard data — 2026-11-16

Total: **1,018 scores**, 35 model labels, 15 languages/scripts, six tasks.

| Task | Scores | Models | Languages/scripts | Empty cells |
|---|---:|---:|---:|---:|
| PIQA | 360 | 24 | 15 | 0 |
| COPA | 230 | 23 | 10 | 0 |

PIQA and COPA use only their current model-by-language result files. All 590
unique scores match those files exactly. Identical duplicate source rows are
imported once. The older incomplete rows previously added to these tasks are no
longer imported. Source files are unchanged.

The other four tasks retain their 428 scores unchanged. Original model IDs remain
in `source_model`. Serbian, Torlak, and Šatrovački labels without script suffixes
denote Latin; Cyrillic results are separate.

Sources:

- `../CLASSLA-PIQA-Benchmark/results/piqa_results.json` (PIQA only source)
- `../DIALECT-COPA-Benchmark/results/copa_results.json` (COPA only source)
- Other tasks retain their `evaluation-for-the-paper/results.json` sources;
  Genre uses `results_combined.json`.

Rebuild: `python3 extract_results.py` (also works via an absolute path from any
directory). Restart or refresh the dashboard after rebuilding.

Validation: 20 unit tests pass, including a regression test preventing older
models or cells from being merged into PIQA/COPA. Exact source comparison,
unchanged other-task data, and byte-identical regeneration pass. Streamlit AppTest
renders both task tables with zero empty cells and no application exceptions.
