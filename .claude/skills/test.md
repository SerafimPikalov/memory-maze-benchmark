---
name: test
description: "Run the test suite and report results"
user_invocable: true
---

Run the project test suite and report results clearly.

1. Run `make test` (skips slow integration tests)
2. If all pass, report: "All tests pass (smoke + environment + training components)"
3. If any fail, show the failure details and suggest fixes
4. Ask if the user wants to run slow integration tests too: `make test-all` (adds a real 200-step training run, ~60s)

Test files:
- `tests/test_smoke.py` — imports, model instantiation (~2s)
- `tests/test_environment.py` — MuJoCo env creation, stepping, wrappers (~10s)
- `tests/test_training.py` — forward pass, losses, V-trace, learn() (~30s)
- `tests/test_training.py -m slow` — real 200-step IMPALA training (~60s)
