#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="readpaper_daily"

# 如已存在旧会话，先杀掉，避免叠加
if /usr/bin/tmux has-session -t "${SESSION}" 2>/dev/null; then
  /usr/bin/tmux kill-session -t "${SESSION}"
fi

# 在一个新的 tmux 会话里运行 run_daily.sh
/usr/bin/tmux new-session -d -s "${SESSION}" "/bin/bash -lc 'cd \"${PROJECT_DIR}\" && bash \"${PROJECT_DIR}/run_daily.sh\"'"
