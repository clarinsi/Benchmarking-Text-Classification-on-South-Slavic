# Interactive Dashboard

The dashboard is available at https://www.clarin.si/classla-llm-dashboard/

## Instructions for dashboard manager

### Updating PIQA and COPA results

The dashboard reads `results.csv`. To rebuild it from the current result files,
run this command from the repository root (or use an absolute script path):

```bash
python3 Interactive-Dashboard/extract_results.py
```

The importer reads the reviewed score matrices directly:

- `CLASSLA-PIQA-Benchmark/results/piqa_results.json`
- `DIALECT-COPA-Benchmark/results/copa_results.json`

These matrices are the only sources for PIQA and COPA. Older
`evaluation-for-the-paper/results.json` files are not merged into these tasks:
their additional incomplete model rows caused empty heatmap cells. PIQA and
COPA include all scripts: Serbian, Torlak, and Šatrovački without a script suffix
mean Latin, with Cyrillic results in separate columns. The other four tasks keep
their existing data sources and filters.

Identical duplicate cells are imported once. Conflicting duplicate values,
unknown language codes, invalid scores, or missing source files stop generation
before the CSV is overwritten. Original model IDs are retained in `source_model`.
Provider aliases such as `google/gemma3:27b` are normalized to the existing display
name. Distinct DeepSeek and GaMS identifiers are not combined across tasks.

The corrected 2026-11-16 refresh contains 360 PIQA scores (24 models × 15
languages/scripts) and 230 COPA scores (23 models × 10 languages), with no empty
cells in either task. The complete CSV contains 1,018 scores across six tasks.
See `current_dashboard_data.md`.

After rebuilding, refresh the local dashboard. For the hosted dashboard, copy
`results.csv` and `app.py` to the server and restart its service. Keep the updated
`extract_results.py` and source matrices together for subsequent rebuilds.

### Importing runs from the separate Slavic benchmark repository

The existing `../sync_benchmark_results.py` script updates the legacy raw/paper
files and rebuilds the dashboard. It does **not** edit the reviewed matrices above
or update the PIQA/COPA scores shown by the dashboard. To publish revised scores,
update the matrix files and run `extract_results.py` directly.

### Checking the importer

```bash
python3 -m unittest discover -s Interactive-Dashboard -p 'test_*.py' -v
```

### Running the dashboard through a service manager (for the CLARIN.SI website)

Option 1 (recommended): `bash /home/tajak/Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard/re-run-service.sh` (a script that contains all the steps listed below)

Option 2 (Step-by-step): run the following commands (from the initial working directory (`\`)):
- assure that no service is running, disable any running service: `systemctl --user disable interactive_dashboard.service` and reload: `systemctl --user daemon-reload`
- enable the service: `systemctl --user enable interactive_dashboard.service`
- start the service: `systemctl --user start interactive_dashboard.service`
- to check the status:
	- `systemctl --user status interactive_dashboard.service` 
	- or `journalctl --user -xeu interactive_dashboard.service`
- to stop the service: `systemctl --user stop interactive_dashboard`

You can find error messages in `Interactive-Dashboard/app.error` and log messages in `Interactive-Dashboard/app.log`

### Running the dashboard locally

Use the `interactive_dash` conda environment.
Or, create a new conda environment and install `pip install streamlit pandas plotly`

To run the dashboard:
- move to the Interactive-Dashboard directory: `Benchmarking-Text-Classification-on-South-Slavic/Interactive-Dashboard`
- first, create a screen: `screen -S classla-dashboard`
- then activate the conda environment: `conda activate interactive_dash`
- then run the streamlit app: `streamlit run app.py > app.log`
- detach from the screen: CTRL + A + D

To re-attach to the screen: `screen -r classla-dashboard` (to close the screen: CTRL + A + D)

Server information:
- port = 48883
- address = "0.0.0.0"
