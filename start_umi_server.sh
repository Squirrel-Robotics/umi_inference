#!/usr/bin/env bash
set -euo pipefail

# Friendly stable entry point. The checkpoint is the only task-specific input.
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/start_umi_v2_server.sh" "$@"
