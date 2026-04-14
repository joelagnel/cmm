Find proven diagnostic strategies for a problem type. Given a description of the problem, returns debugging approaches that worked in past sessions on this project.

Run the following command with the user's problem description and present the results:

```bash
CMM_PROJECT_ID="${CMM_PROJECT_ID}" CMM_STORE_PATH="${CMM_STORE_PATH}" "$CMM_PYTHON" -m src.delivery.cli_query diagnose "$ARGUMENTS"
```

After presenting the results, suggest which strategy seems most applicable and outline the first steps to take.
