#!/usr/bin/env bash
set -euo pipefail

STRATEGY=per_sample_coverage_similarity \
METHOD_KEY=per_sample_coverage_similarity20x7 \
bash "$(dirname "${BASH_SOURCE[0]}")/run_en_coverage_selection20x7_dev_oof.sh"
