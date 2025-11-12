  # Install UV
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Create environment and install dependencies
  uv sync

  # Activate environment
  source .venv/bin/activate  # or uv run for individual commands

  # Run analysis
  uv run python scripts/hippocampus/run_bandit_analysis.py

  # Install in development mode
  uv pip install -e .