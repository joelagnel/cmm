Generate an interactive HTML visualization of the reasoning DAG for this project. Opens in any browser with animated nodes, color-coded by type, clickable details, and zoom/pan.

Run the following command and tell the user where the output file is:

```bash
cd "$CMM_ROOT" && source "$CMM_ENV" && "$CMM_PYTHON" scripts/visualize_dag.py -p "${CMM_PROJECT_ID}" --store "${CMM_STORE_PATH}" -o output/ -f html
```

After running, tell the user to open the generated HTML file in their browser.
