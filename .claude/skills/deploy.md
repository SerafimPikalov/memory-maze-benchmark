---
name: deploy
description: "Build the Docker image (and optionally push to Docker Hub)"
user_invocable: true
---

Build the Docker image from the current repo state.

1. Ask the user: "Build locally only, or also push to Docker Hub?"
2. If build only: `./docker/deploy.sh`
3. If push: ask for their Docker Hub username, then: `DOCKER_REPO=username/memorymaze-train ./docker/deploy.sh`

Notes:
- Building takes ~10 min (downloads CUDA base image, installs Genesis, patches gs-madrona)
- Requires `docker` with buildx support
- On Apple Silicon, builds with `--platform linux/amd64` (cross-compilation, slower)
- The Docker Hub image `serapikalov/memorymaze-train` may be stale — building from source is recommended
- Never push to `serapikalov/memorymaze-train` unless the user explicitly owns that account
