#!/usr/bin/env bash
set -Eeuo pipefail

tmux new-session -d -s rachel_jupyter "jupyter lab"
