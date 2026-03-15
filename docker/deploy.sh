#!/usr/bin/env bash
# Build and push the Memory Maze training Docker image.
#
# Usage:
#   ./docker/deploy.sh                                # build + push (uses DOCKER_REPO)
#   ./docker/deploy.sh --build-only                   # build without pushing
#   DOCKER_REPO=myuser/myimage ./docker/deploy.sh     # build + push to custom repo
#
# If DOCKER_REPO is not set, the script asks whether to push.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_ONLY=0
if [ "${1:-}" = "--build-only" ]; then
    BUILD_ONLY=1
fi

# Default Docker repo from pod_manager.py
DEFAULT_REPO="serapikalov/memorymaze-train"

if [ -z "${DOCKER_REPO:-}" ]; then
    DOCKER_REPO="$DEFAULT_REPO"
    if [ "$BUILD_ONLY" = "0" ]; then
        echo "DOCKER_REPO not set. Default: ${DEFAULT_REPO}"
        echo ""
        echo "  1) Build and push to ${DEFAULT_REPO} (RunPod will use this)"
        echo "  2) Build only (local image, RunPod won't see it)"
        echo ""
        read -p "Choice [1/2]: " choice
        if [ "${choice:-2}" = "2" ]; then
            BUILD_ONLY=1
            DOCKER_REPO="memorymaze-train"
        fi
    fi
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

echo ""
echo "Build complete: ${IMAGE}"

if [ "$BUILD_ONLY" = "0" ]; then
    echo "Pushing ${IMAGE}..."
    docker push "$IMAGE"
    echo ""
    echo "Push complete. RunPod pods will now use this image."
else
    echo ""
    echo "NOTE: This image is LOCAL only."
    echo "  - 'docker run --gpus all ${IMAGE}' works on this machine"
    echo "  - RunPod pods still pull from Docker Hub (may be stale)"
    echo ""
    echo "To push for RunPod:"
    echo "  DOCKER_REPO=${DEFAULT_REPO} ./docker/deploy.sh"
fi

echo "============================================================"
