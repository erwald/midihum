#!/usr/bin/env bash
set -Eeuo pipefail

tmux new-session -d -s midihum_jupyter "jupyter lab"
