# DUCKDB SQL DEMOS

Here I am running python script to analyse data and visualize them with sql template.yml and visualize with report_template.html 

## Quick start
```bash
uv run main.py --db_path load_statistics.db
```

# Fast startup with validation only
```bash
uv run main.py --db_path load_statistics.db --validate-only
```
# Full analysis with performance options
```bash
uv run main.py --db_path load_statistics.db --parallel --threads 8
```

# Fastest analysis without visualizations (data only)
```bash
uv run main.py --db_path load_statistics.db --skip-viz --parallel
```

## For Testing:
```bash
tests.ipynb
```