#!/usr/bin/env bash
# Compatibility entry point: existing sbatch commands keep their defaults/ports.
set -euo pipefail
exec bash "${H3_DEPLOY_ROOT:-/home/jeff/sol-h3-h200}/Sol-H3/deploy_node.sh" "$@"
