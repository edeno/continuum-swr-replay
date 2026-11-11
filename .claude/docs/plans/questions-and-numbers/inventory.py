"""Inventory for the questions-and-numbers design.

Parses paper/claims_table.md into rows (part heading, claim id, cells verbatim) and joins
each claim id to the values in paper/reported/part{n}.json that name it. Prints JSON.
Run from the repository root: python3 <this file>.
"""

import collections
import json
import re
from pathlib import Path

root = Path.cwd()
rows = []
part = None
for lineno, line in enumerate(
    (root / "paper/claims_table.md").read_text().splitlines(), 1
):
    if not line.startswith("|"):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    m = re.match(r"\*\*(Part [0-9][a-d]?) ", cells[0])
    if m:
        part = m.group(1)
        continue
    if re.fullmatch(r"[0-9]+[a-g]?", cells[0]):
        rows.append(
            {
                "line": lineno,
                "part": part,
                "id": cells[0],
                "claim": cells[1],
                "comparison": cells[2],
                "placement": cells[-1],
            }
        )

values = collections.defaultdict(list)
for n in range(5):
    data = json.loads((root / f"paper/reported/part{n}.json").read_text())["values"]
    for key, v in data.items():
        for c in v["claims"]:
            values[c].append((n, key, v["used_in"]))

out = []
for r in rows:
    vs = values.get(r["id"], [])
    tags = collections.Counter(u for _, _, used in vs for u in used)
    out.append(
        {
            **r,
            "n_values": len(vs),
            "summary_parts": sorted({n for n, _, _ in vs}),
            "used_in": dict(tags),
        }
    )

table_ids = {r["id"] for r in rows}
print(
    json.dumps(
        {
            "rows": out,
            "claims_in_values_not_in_table": sorted(set(values) - table_ids),
            "duplicate_ids": [
                i
                for i, c in collections.Counter(r["id"] for r in rows).items()
                if c > 1
            ],
        },
        indent=1,
    )
)
