Start the cognitive memory session watcher daemon. This watches ~/.claude/projects/ for new or modified session files and automatically ingests them using warm-tier heuristic extraction.

Run the following command:

```bash
cd "$CMM_ROOT" && source "$CMM_ENV" && "$CMM_PYTHON" -m src.ingestion.watcher --store "${CMM_STORE_PATH}"
```

After starting, the watcher runs continuously. Tell the user it's watching for new sessions and will auto-ingest them. They can stop it with Ctrl+C.
