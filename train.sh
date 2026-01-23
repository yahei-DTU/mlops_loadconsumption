#!/bin/bash
set -e

export DVC_NO_SCM=1

uv run dvc pull
uv run python src/mlops_loadconsumption/train.py
