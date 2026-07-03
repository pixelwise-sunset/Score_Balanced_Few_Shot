#!/usr/bin/env bash
set -euo pipefail

STRATEGIES="${STRATEGIES:-metric_bin_coverage20x7 per_sample_coverage_similarity20x7}" \
bash "$(dirname "${BASH_SOURCE[0]}")/run_en_shot_selection_probabilistic20x7.sh"
