# Development Scratchpad

**Last Updated:** 2025-11-14
**Current Task:** CB-3.1 - Use `|` operator for dict merging

## Session Notes

### 2025-11-14: CB-3.1 - BLOCKED (Mypy Type Inference Limitation)

**Task:** Replace `{**dict1, **dict2}` with `dict1 | dict2` in bandit_task.py (lines 181-192)

**Status:** BLOCKED - Discovered mypy limitation with `|` operator and TypedDict

**Investigation Summary:**
1. ✅ Checked existing tests - 20 integration tests cover `load_data()` structure
2. ✅ Ran baseline tests - all 20 tests pass with original code
3. ✅ Implemented `|` operator refactor
4. ✅ Tests still pass (runtime behavior identical)
5. ❌ Mypy fails with `|` operator: "Incompatible return value type (got 'dict[str, object]', expected 'SessionData')"
6. ✅ Verified old `{**dict}` syntax passes mypy without issues

**Root Cause:**
- The `|` operator for dict merging returns `dict[str, object]` type
- This is incompatible with the `SessionData` TypedDict return type
- The `{**dict}` unpacking syntax has BETTER type inference for TypedDict merging
- This is a known mypy limitation, not a bug in our code

**Mypy Behavior Comparison:**
- Old syntax `{**position_data, **spike_data, ...}`: mypy correctly infers `SessionData` type ✅
- New syntax `position_data | spike_data | ...`: mypy infers `dict[str, object]` ❌

**Attempted Solutions:**
- Used `cast("SessionData", ...)` - works but user rejected (code smell)
- No other solutions available without sacrificing type safety

**Decision Required:**
1. Keep old `{**dict}` syntax (better mypy support for TypedDict) ← **RECOMMENDED**
2. Mark CB-3.1 as NOT APPLICABLE (similar to CB-1.3)

**Recommendation:** Mark CB-3.1 as NOT APPLICABLE - The `{**dict}` syntax is actually the CORRECT and RECOMMENDED approach for TypedDict merging due to superior type inference in mypy. The `|` operator is great for general dict merging but has limitations with TypedDicts.

---

## Completed Work

### Milestone 1: Critical Fixes ✅
- FU-1.1 through FU-1.5: Figure utilities critical fixes
- CB-1.1 through CB-1.2: Codebase critical fixes
- CB-1.3: Marked NOT APPLICABLE (enumerate() is correct)

### Milestone 2: UX Critical Fixes ✅
- FU-2.1 through FU-2.5: All UX improvements completed

---

## Blockers

None currently.

---

## Decisions Log

### CB-1.3: enumerate() with itertuples()
**Decision:** NOT APPLICABLE - enumerate() is correct here
**Rationale:** DataFrame index may be non-sequential, while labels must be sequential
**Evidence:** Tests failed when using row.Index instead of enumerate counter

### CB-3.1: Use | operator for dict merging
**Decision:** NOT APPLICABLE - {**dict} is correct for TypedDict merging
**Rationale:** The | operator has inferior type inference for TypedDict merging in mypy
**Evidence:**
  - `{**dict}` syntax: mypy correctly infers SessionData type
  - `dict | dict` syntax: mypy infers dict[str, object], incompatible with SessionData
  - Using cast() is a code smell and indicates wrong approach
  - This is a documented mypy limitation, not a bug
**Action Taken:** Removed unused `Any` import as cleanup
