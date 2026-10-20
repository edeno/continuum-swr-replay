# Test Suite Quick Start

```bash
uv sync --extra dev
uv run pytest -q -p no:cacheprovider   # full suite; real-data tests skipped
uv run pytest -m real_data             # the real-data tests (needs Processed-Data, the VPN)
```

Everything else — the `real_data`, `integration` and `slow` markers, the layout,
fixtures and conventions — is in [README.md](README.md).
