Run batch consolidation for this project — rebuilds the cognitive profile from all stored reasoning nodes. Optionally upgrades warm-tier nodes with full LLM extraction.

Run the following command and present the results:

```bash
cd "$CMM_ROOT" && source "$CMM_ENV" && "$CMM_PYTHON" scripts/batch_consolidate.py -p "${CMM_PROJECT_ID}" --store-dir "${CMM_STORE_PATH}" --profiles-only
```

After presenting results, summarize what changed in the cognitive profile (new insights, pitfalls, or strategies discovered).
