#!/usr/bin/env bash
set -euo pipefail

STRATEGY=metric_bin_coverage \
METHOD_KEY=metric_bin_coverage20x7 \
bash "$(dirname "${BASH_SOURCE[0]}")/run_en_coverage_selection20x7_dev_oof.sh"
