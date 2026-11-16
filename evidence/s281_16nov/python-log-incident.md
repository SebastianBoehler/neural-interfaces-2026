# Python executable log-append incident — 16 Nov 2026

`run_capped.sh` expects `LOG CMD [ARGS...]`. It was accidentally invoked with `.venv/bin/python` as `LOG` and `-I` as `CMD`. The resolved log target was `/Users/sebastianboehler/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12`. The command did not score anything: the shell could not find command `-I`.

The following exact 201 bytes were appended, including the final newline:

```text
[run_capped] Mon Nov 16 15:13:41 CET 2026 start: -I /tmp/ni26_score_grid_recovery.py (max 8 GB)
scripts/run_capped.sh: line 28: -I: command not found
[run_capped] Mon Nov 16 15:13:51 CET 2026 exit 127
```

The suffix was checked for exact byte equality before truncating precisely 201 bytes. The resulting executable size is 18,073,888 bytes. `codesign --verify --verbose=2 .venv/bin/python` reported `valid on disk` and `satisfies its Designated Requirement`. The restored interpreter ran and imported NumPy: Python 3.12.13, NumPy 2.5.4. No other environment change was made during restoration.
