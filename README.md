  # Install UV
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Create environment and install dependencies
  uv sync

  # Activate environment
  source .venv/bin/activate  # or uv run for individual commands

  # Run decoding analysis (see scripts/run_decoding_bandit.py --help)
  uv run python scripts/run_decoding_bandit.py --animal chimi --brain_area HPC

  # Install in development mode
  uv pip install -e .