#!/usr/bin/env bash
# Build and push the Memory Maze training Docker image.
#
# Usage:
#   ./docker/deploy.sh                                # build only (no DOCKER_REPO set)
#   ./docker/deploy.sh --build-only                   # build without pushing
#   DOCKER_REPO=myuser/myimage ./docker/deploy.sh     # build + push to your repo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_ONLY=0
if [ "${1:-}" = "--build-only" ]; then
    BUILD_ONLY=1
fi

if [ -z "${DOCKER_REPO:-}" ]; then
    echo "DOCKER_REPO not set — building locally as 'memorymaze-train' (no push)."
    DOCKER_REPO="memorymaze-train"
    BUILD_ONLY=1
fi

TAG="${TAG:-latest}"
IMAGE="${DOCKER_REPO}:${TAG}"

echo "============================================================"
echo "Building Docker image"
echo "  Image: ${IMAGE}"
echo "  Context: ${REPO_ROOT}"
echo "  Dockerfile: docker/Dockerfile"
echo "  Platform: linux/amd64"
echo "============================================================"

cd "$REPO_ROOT"

docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    -t "$IMAGE" \
    .

echo "Build complete: ${IMAGE}"

if [ "$BUILD_ONLY" = "0" ]; then
    echo "Pushing ${IMAGE}..."
    docker push "$IMAGE"
    echo "Push complete."
fi

echo "============================================================"
echo "Done. Image: ${IMAGE}"
echo "============================================================"
