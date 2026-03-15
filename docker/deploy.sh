#!/usr/bin/env bash
# Build and push the Memory Maze training Docker image.
#
# Usage:
#   ./docker/deploy.sh                          # build + push
#   ./docker/deploy.sh --build-only             # build without pushing
#   DOCKER_REPO=myuser/myimage ./docker/deploy.sh  # custom repo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DOCKER_REPO="${DOCKER_REPO:-serapikalov/memorymaze-train}"
TAG="${TAG:-latest}"
IMAGE="${DOCKER_REPO}:${TAG}"

BUILD_ONLY=0
if [ "${1:-}" = "--build-only" ]; then
    BUILD_ONLY=1
fi

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
