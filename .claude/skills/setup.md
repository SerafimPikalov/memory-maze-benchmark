---
name: setup
description: "Install dependencies and verify the project works"
user_invocable: true
---

Set up the Memory Maze benchmark project. Detect the environment and follow the right path:

1. Check platform: `uname -s` (Darwin = macOS, Linux = headless server likely)
2. Install dependencies: `pip install -r requirements.txt`
3. Run fast tests: `make test`
4. Report results: what passed, what failed, what to do next

If tests pass, suggest:
- macOS: "Try `cd notebooks && jupyter lab` to explore interactively"
- Linux with GPU: "Try `make benchmark-backends` to compare MuJoCo vs Genesis"
- Linux no GPU: "Tests pass! For GPU training, see `make train` or set up a RunPod with the runpod-manager agent"
