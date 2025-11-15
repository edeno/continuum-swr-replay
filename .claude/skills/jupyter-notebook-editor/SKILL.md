---
name: jupyter-notebook-editor
description: Edit Jupyter notebooks reliably using jupytext paired mode. This skill should be used when users want to modify, refactor, or fix code in .ipynb files. The skill sets up automatic synchronization between notebooks and Python scripts for reliable editing.
---

# Jupyter Notebook Editor

Edit Jupyter notebooks reliably using jupytext paired mode, which maintains automatic synchronization between `.ipynb` notebooks and `.py` Python scripts.

## Overview

Users may request edits, modifications, or refactoring of Jupyter notebook (.ipynb) code. Jupyter notebooks are JSON files with complex nested structures—direct JSON editing is error-prone and can corrupt notebooks. This skill uses jupytext paired mode, the industry-standard approach from the Jupyter community, to maintain synchronized `.ipynb` and `.py` files for reliable editing.

## When to Use This Skill

**Use this skill when users request to:**

- Edit, modify, or refactor code in Jupyter notebooks
- Fix bugs or errors in notebook cells
- Add, remove, or reorganize notebook cells
- Improve code quality in notebooks (add docstrings, rename variables, refactor structure)
- Make structural changes to notebook flow

**Do NOT use this skill for:**

- Running or executing notebooks → Use `jupyter nbconvert --execute notebook.ipynb`
- Viewing notebook content → Use `python -m markitdown notebook.ipynb`
- Creating new notebooks → Create .ipynb directly with Jupyter
- Debugging runtime errors → Execute in Jupyter to see errors
- Installing packages → Add `!pip install` commands in notebook cells

## Prerequisites

**CRITICAL**: Verify jupytext is installed before proceeding.

Check installation:

```bash
jupytext --version
```

**If command fails**, jupytext is not installed. Inform the user:
> "Jupytext is required but not installed. Please run: `pip install jupytext`"

**Do not proceed** without jupytext installed.

## Quick Reference

**Decision tree:**

```
1. Check if notebook is paired
   └─ jupytext --show-formats notebook.ipynb
      ├─ Output: "ipynb,py:percent" → Already paired, skip to Step 3
      └─ Output: "ipynb" → Not paired, continue to Step 2

2. Set up pairing (if needed)
   └─ jupytext --set-formats ipynb,py:percent notebook.ipynb

3. Edit the .py file
   └─ Make requested changes

4. Sync changes
   └─ jupytext --sync notebook.ipynb

5. Validate
   └─ Verify both files exist and notebook structure is valid
```

**Key rules:**

- Always check pairing status first (don't assume)
- Always edit the `.py` file, never edit `.ipynb` directly
- Cell markers: `# %%` (code cells) and `# %% [markdown]` (markdown cells)
- Sync after editing to update the notebook

## Workflow

Follow these steps in order. Do not skip steps.

### Step 1: Check Pairing Status

**Before any editing**, check if the notebook is already paired:

```bash
jupytext --show-formats notebook.ipynb
```

**Expected outputs:**

- Already paired: `ipynb,py:percent`
- Not paired: `ipynb`

**If already paired** → Skip to Step 3 (Edit)
**If not paired** → Continue to Step 2 (Set up pairing)

**If command fails:**

- Verify notebook file exists: `ls notebook.ipynb`
- Check notebook is valid JSON: `python -c "import json; json.load(open('notebook.ipynb'))"`
- Report error to user

### Step 2: Set Up Pairing (Only If Not Already Paired)

**Only execute this step** if Step 1 showed notebook is not paired.

Set up automatic synchronization between `.ipynb` and `.py`:

```bash
jupytext --set-formats ipynb,py:percent notebook.ipynb
```

**Expected result:** Creates `notebook.py` alongside `notebook.ipynb`

**Verify pairing succeeded:**

```bash
ls notebook.py  # Should exist
jupytext --show-formats notebook.ipynb  # Should show: ipynb,py:percent
```

**If setup fails:**

- Verify write permissions in directory
- Check notebook file is not corrupted
- Ensure notebook name doesn't contain special characters
- Report error to user

**Note:** Pairing configuration is stored in notebook metadata and persists. This step only needs to be done once per notebook.

### Step 3: Edit the Python Script

**CRITICAL**: Edit the `.py` file, never edit the `.ipynb` file directly.

The `.py` file uses percent format with cell markers:

```python
# %%
# This is a code cell
import pandas as pd

# %% [markdown]
# # This is a markdown cell
# Markdown content appears as comments

# %%
# Another code cell
df = pd.read_csv('data.csv')
```

**Editing guidelines:**

- Add new cells by adding `# %%` markers
- Remove cells by deleting content between `# %%` markers
- Edit code within cells normally
- Modify markdown cells as comments after `# %% [markdown]` markers
- Cell markers must be exactly `# %%` (with space)

**Make the requested changes** to the `.py` file.

### Step 4: Sync Changes to Notebook

After editing the `.py` file, sync changes back to the `.ipynb` file:

```bash
jupytext --sync notebook.ipynb
```

**Expected outputs:**

- Changes detected: `[jupytext] Updating 'notebook.ipynb'`
- No changes: `[jupytext] Notebooks are up to date`

**How sync works:** Jupytext automatically detects which file (`.py` or `.ipynb`) was modified more recently and updates the other. The command is always the same regardless of which file was edited.

**If sync fails:**

- Check both files exist: `ls notebook.ipynb notebook.py`
- Verify `.py` file has valid syntax
- Check for malformed cell markers
- Look for error message indicating the issue
- Report error to user

### Step 5: Validate the Result

**CRITICAL**: Verify the notebook after syncing.

**Required validation checks:**

1. **Both files exist:**

   ```bash
   ls notebook.ipynb notebook.py
   ```

2. **Notebook has valid JSON structure:**

   ```bash
   python -c "import json; json.load(open('notebook.ipynb'))"
   ```

   Expected: No error output

3. **Notebook opens in Jupyter:**

   ```bash
   jupyter notebook notebook.ipynb
   ```

**If validation fails:**

- Review `.py` file for syntax errors
- Check cell markers are properly formatted
- Verify no content was accidentally deleted
- Re-edit and re-sync if needed

## Command Reference

### Check Pairing Status

```bash
jupytext --show-formats notebook.ipynb
```

**Output if paired:** `ipynb,py:percent`
**Output if not paired:** `ipynb`

### Set Up Pairing

```bash
jupytext --set-formats ipynb,py:percent notebook.ipynb
```

**Success:** No output, creates `.py` file
**Failure:** Error message about file access or invalid notebook

### Sync Changes

```bash
jupytext --sync notebook.ipynb
```

**Changes synced:** `[jupytext] Updating 'notebook.ipynb'` or `[jupytext] Updating 'notebook.py'`
**No changes:** `[jupytext] Notebooks are up to date`
**Failure:** Error message about conflicts or parsing errors

### Force Sync Direction (Advanced)

```bash
# Force .py → .ipynb
jupytext --sync --to ipynb notebook.py

# Force .ipynb → .py
jupytext --sync --to py:percent notebook.ipynb
```

## Example Workflows

### Example 1: Fix a Bug

**User request:** "Fix the division by zero bug in analysis.ipynb cell 3"

**Actions:**

1. Check pairing: `jupytext --show-formats analysis.ipynb`
   - If not paired → Set up: `jupytext --set-formats ipynb,py:percent analysis.ipynb`
2. Edit `analysis.py` → Navigate to third cell (third `# %%` block), add zero check
3. Sync: `jupytext --sync analysis.ipynb`
4. Validate: Verify notebook opens and contains fix

**Expected result:** Bug fixed, both files updated

### Example 2: Add New Cell

**User request:** "Add a cell that creates a scatter plot"

**Actions:**

1. Check pairing: `jupytext --show-formats plotting.ipynb`
   - If not paired → Set up: `jupytext --set-formats ipynb,py:percent plotting.ipynb`
2. Edit `plotting.py` → Add at appropriate location:

   ```python
   # %%
   import matplotlib.pyplot as plt
   plt.scatter(df['x'], df['y'])
   plt.xlabel('X values')
   plt.ylabel('Y values')
   plt.show()
   ```

3. Sync: `jupytext --sync plotting.ipynb`
4. Validate: Verify notebook has new cell

**Expected result:** New cell added to notebook

### Example 3: Refactor Variable Names

**User request:** "Refactor data_cleaning.ipynb to use better variable names"

**Actions:**

1. Check pairing: `jupytext --show-formats data_cleaning.ipynb`
   - If not paired → Set up: `jupytext --set-formats ipynb,py:percent data_cleaning.ipynb`
2. Edit `data_cleaning.py` → Refactor variables (e.g., `df1` → `sales_data`)
3. Sync: `jupytext --sync data_cleaning.ipynb`
4. Validate: Execute to ensure refactoring didn't break code:

   ```bash
   jupyter nbconvert --to notebook --execute data_cleaning.ipynb
   ```

**Expected result:** Variables renamed, code still functional

## Troubleshooting

### Pairing Not Set Up

**Symptom:** No `.py` file exists or sync fails with "file not found"

**Solution:** Set up pairing first:

```bash
jupytext --set-formats ipynb,py:percent notebook.ipynb
```

**Verify:** Check `.py` file was created: `ls notebook.py`

### Cell Markers Not Recognized

**Symptom:** Cells missing or merged after sync

**Check:** Cell markers are exactly `# %%` with space

- ✓ Correct: `# %%`
- ✗ Wrong: `#%%` (no space)
- ✗ Wrong: `##%%` (double hash)

**Check:** Markdown markers are `# %% [markdown]`

**Check:** Blank lines between cells weren't removed

### Changes Not Syncing

**Symptom:** Edit `.py` file but `.ipynb` doesn't update

**Check:** Both files exist: `ls notebook.ipynb notebook.py`

**Check:** Sync command ran without errors: `jupytext --sync notebook.ipynb`

**Force sync from .py:** `jupytext --sync --to ipynb notebook.py`

### Jupytext Command Not Found

**Symptom:** `bash: jupytext: command not found`

**Solution:** Inform user jupytext must be installed:
> "Jupytext is not installed. Please run: `pip install jupytext`"

**Do not proceed** without jupytext.

### Syntax Errors in .py File

**Symptom:** Sync fails with parsing error

**Check:** Python syntax in `.py` file is valid

**Check:** Cell markers are properly formatted

**Solution:** Fix syntax errors in `.py` file and sync again

## Important Notes

- **Pairing is persistent:** Once set up, pairing configuration is stored in notebook metadata
- **Automatic sync direction:** Jupytext syncs from the most recently modified file
- **Output preservation:** Cell outputs in `.ipynb` are preserved during sync (not stored in `.py`)
- **Edit the .py file:** Always edit `.py` for reliability; editing `.ipynb` JSON directly can corrupt the notebook
- **Version control:** Commit both `.ipynb` and `.py` files to git for best collaboration
- **Better diffs:** `.py` files have readable git diffs compared to `.ipynb` JSON

## Benefits of Paired Mode

1. **One-time setup** - Configure pairing once, use forever
2. **Automatic synchronization** - Edit either file, changes sync to both
3. **Industry standard** - Best practice from Jupyter community
4. **Version control friendly** - Readable diffs on `.py` files
5. **IDE support** - Full IDE features when editing `.py` files
6. **Team collaboration** - Share both files in git for seamless workflows

## Dependencies

**Required:**

- **jupytext** - `pip install jupytext` (notebook-script synchronization)
- **jupyter** - `pip install jupyter` (opening/verifying notebooks)

**Optional:**

- **nbconvert** - `pip install nbconvert` (format conversion, execution)
- **nbformat** - `pip install nbformat` (programmatic manipulation)

## References

- Jupytext documentation: <https://jupytext.readthedocs.io/>
- Paired notebooks: <https://jupytext.readthedocs.io/en/latest/paired-notebooks.html>
- Percent format: <https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format>
