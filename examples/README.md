# Example Configurations

These samples show common Scrapi Reddit setups. Copy one to your project directory and tweak the values before running `scrapi-reddit --config <file>`.

| File | Purpose |
| --- | --- |
| `basic-subreddits.toml` | Grab popular/hot posts from a few subreddits with CSV exports. |
| `keyword-search.toml` | Site-wide keyword search that downloads comments and media. |
| `geo-popular.toml` | Monitor r/popular with geo filters and time windows. |
| `user-activity.toml` | Track a user's submitted, overview, and comments listing. |
| `mixed-run.toml` | Combo run mixing subreddits, keyword search, and saved posts. |
| `run_python_api.py` | Script version of the Python API example that writes outputs to `./example_runs`. |

Run any of them like:

```powershell
scrapi-reddit --config examples/basic-subreddits.toml
```

Override values on the command line (flags always win):

```powershell
scrapi-reddit --config examples/basic-subreddits.toml --limit 50 --output-format json
```
