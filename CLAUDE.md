# CLAUDE.md

@docs/development.md

The project's rules live in [docs/development.md](docs/development.md), imported
above, so that people and the assistant work from one copy. Edit them there, not
here. This file holds only what applies to the assistant alone.

- Never write to the lab's Spyglass database (insert, populate, delete, schema
  creation, export) without the user's explicit approval of that specific step; the
  analyses only read it (see "Data Integration with Spyglass" in the guide).
- A result number stated in a chat reply follows the same rule as one in a doc:
  read it from a summary file, `paper/reported/part{n}.json`, and name the file
  and the key. If no summary holds it, say it is not computed (see "Reported
  numbers" in the guide).
