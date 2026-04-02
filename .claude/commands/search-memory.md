Search cognitive memory for reasoning patterns relevant to a problem. Use this when you hit an unfamiliar error, need to understand a subsystem, or want to know if a similar problem has been solved before.

Run the following command with the user's query and present the results:

```bash
CMM_PROJECT_ID="${CMM_PROJECT_ID}" CMM_STORE_PATH="${CMM_STORE_PATH}" cd "/Users/sazankhalid/Downloads/cmm/cognitive-memory" && source "/Users/sazankhalid/Downloads/cmm/.claude/cmm-env.sh" && "$CMM_PYTHON" -m src.delivery.cli_query search "$ARGUMENTS"
```

After presenting the results, summarize the most relevant findings and how they apply to the current situation.
