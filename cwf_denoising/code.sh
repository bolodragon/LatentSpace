#!/usr/bin/env bash
# Unified local / Slurm launcher for code.py.
# Put both files in mlls/, or set CRYO_PROJECT_DIR to your data directory.
# Original Dardel account/partition are retained below; override if needed.
#
# Local:
#   bash code.sh check
#   bash code.sh all
#   bash code.sh cwf --limit 256
#   bash code.sh halo --input /path/particles_cwf.mrc --limit 100
# Slurm (two jobs, halo waits for successful CWF):
#   bash code.sh submit-all
#   bash code.sh submit-cwf --limit 256
#   bash code.sh submit-halo --input /path/particles_cwf.mrc --limit 100
# Optional original tests/pilot:
#   bash code.sh test --reference-stack result/cwf_26472411/particles_cwf.mrc
#   bash code.sh pilot --input result/cwf_26472411/particles_cwf.mrc
#
# One-time setup, if no existing environment:
#   python3 -m venv .aspire-env
#   .aspire-env/bin/python code.py requirements | .aspire-env/bin/python -m pip install -r /dev/stdin
# The pins are the recorded environment, not a claim of current availability.
# No packages are installed by this launcher.
#
# Output directories must be fresh. Full runs also write eight-orientation
# vectors (about 7 GiB for 57,107 particles); see code.py comments for limitations.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${CRYO_PROJECT_DIR:-$SCRIPT_DIR}"
export CRYO_PROJECT_DIR="$PROJECT_DIR"
PYTHON="${CRYO_PYTHON:-$PROJECT_DIR/.aspire-env/bin/python}"
ACCOUNT="${SLURM_ACCOUNT:-naiss2025-3-71}"
PARTITION="${SLURM_PARTITION:-main}"
# A submitted job executes this file from Slurm's spool, so keep the real path.
export CRYO_CODE_SCRIPT="${CRYO_CODE_SCRIPT:-$SCRIPT_DIR/code.py}"
export CRYO_LAUNCHER="${CRYO_LAUNCHER:-$SCRIPT_DIR/code.sh}"
export MPLBACKEND=Agg MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

run_python() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun "$PYTHON" -u "$CRYO_CODE_SCRIPT" "$@"
    else
        "$PYTHON" -u "$CRYO_CODE_SCRIPT" "$@"
    fi
}
submit() {
    local stage="$1"; shift
    local cpus=4 duration=00:30:00
    if [[ "$stage" == cwf ]]; then cpus=8; duration=01:00:00; fi
    sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
        --job-name="cryo_$stage" --time="$duration" --nodes=1 --ntasks=1 \
        --cpus-per-task="$cpus" --output="$PROJECT_DIR/cryo_${stage}_%j.log" \
        --error="$PROJECT_DIR/cryo_${stage}_%j.err" \
        "${DEPENDENCY_ARGS[@]}" "$CRYO_LAUNCHER" "$stage" "$@"
}
DEPENDENCY_ARGS=()
mode="${1:---help}"
if (( $# )); then shift; fi
case "$mode" in
    -h|--help|help)
        sed -n '2,/^set -euo pipefail/p' "$CRYO_LAUNCHER" | sed '$d;s/^# *//'
        ;;
    check)
        "$PYTHON" -c "import aspire, mrcfile, numpy, scipy, skimage; print('Imports OK')"
        ;;
    submit-all)
        if (( $# )); then echo "submit-all takes no options; use submit-cwf/submit-halo for customized runs." >&2; exit 2; fi
        cwf_job="$(submit cwf)"
        cwf_job="${cwf_job%%;*}"
        DEPENDENCY_ARGS=("--dependency=afterok:$cwf_job")
        halo_job="$(submit halo --input "$PROJECT_DIR/result/cwf_${cwf_job}/particles_cwf.mrc")"
        printf 'CWF job: %s\nHalo job: %s\n' "$cwf_job" "$halo_job"
        ;;
    submit-cwf|submit-halo|submit-segment)
        submit "${mode#submit-}" "$@"
        ;;
    all)
        if (( $# )); then echo "all takes no options; use individual stages for customized runs." >&2; exit 2; fi
        stamp="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
        cwf_dir="$PROJECT_DIR/result/cwf_$stamp"
        run_python cwf --output-dir "$cwf_dir"
        run_python test
        run_python halo --input "$cwf_dir/particles_cwf.mrc" --output-dir "$PROJECT_DIR/new_test/halo_postprocess_$stamp"
        ;;
    cwf|halo|segment)
        stamp="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
        if [[ "$mode" == cwf ]]; then
            out="$PROJECT_DIR/result/cwf_$stamp"
        else
            export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
            run_python test
            out="$PROJECT_DIR/new_test/${mode}_postprocess_$stamp"
        fi
        # A later user-supplied --output-dir overrides this default.
        run_python "$mode" --output-dir "$out" "$@"
        ;;
    test|pilot|inspect|class-shapes|lowpass|requirements)
        run_python "$mode" "$@"
        ;;
    *)
        echo "Unknown command: $mode (use bash code.sh --help)" >&2
        exit 2
        ;;
esac

# Original launcher/run snippets, preserved for provenance as comments.
# These referred to separate files; use the unified commands above.

# --- Original guide block 2 ---
# cd /cfs/klemming/scratch/p/penotcam/mlls
# .aspire-env/bin/python -c "import aspire, mrcfile, numpy, scipy, skimage; print('Imports OK')"

# --- Original guide block 3 ---
# cd /cfs/klemming/scratch/p/penotcam/mlls
# cwf_job=$(sbatch --parsable cwf_denoise.sh)
# cwf_job=${cwf_job%%;*}
# sbatch --dependency=afterok:"${cwf_job}" new_test/run_halo_postprocess.sh \
#     --input "$PWD/result/cwf_${cwf_job}/particles_cwf.mrc"

# --- Original guide block 4 ---
# cd /cfs/klemming/scratch/p/penotcam/mlls/new_test
# sbatch run_halo_postprocess.sh \
#     --input /cfs/klemming/scratch/p/penotcam/mlls/result/cwf_26472411/particles_cwf.mrc

# --- Original guide block 7 ---
# #!/bin/bash
# #SBATCH -A naiss2025-3-71
# #SBATCH -J cwf_denoise
# #SBATCH -p main
# #SBATCH -t 01:00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --output=/cfs/klemming/scratch/p/penotcam/mlls/cwf_output_%j.log
# #SBATCH --error=/cfs/klemming/scratch/p/penotcam/mlls/cwf_error_%j.log
# 
# set -euo pipefail
# WORK_DIR="/cfs/klemming/scratch/p/penotcam/mlls"
# cd "${WORK_DIR}"
# export MPLBACKEND=Agg
# export MPLCONFIGDIR="${WORK_DIR}/.matplotlib"
# export OPENBLAS_NUM_THREADS=1
# export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# export PYTHONUNBUFFERED=1
# RESULT_DIR="${WORK_DIR}/result/cwf_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
# # One-time setup: python -m venv .aspire-env; .aspire-env/bin/pip install aspire==0.14.3
# srun "${WORK_DIR}/.aspire-env/bin/python" -u "${WORK_DIR}/cwf_denoise.py" \
#     --output-dir "${RESULT_DIR}" "$@"

# --- Original guide block 16 ---
# #!/bin/bash
# #SBATCH -A naiss2025-3-71
# #SBATCH -J notion_postprocess
# #SBATCH -p main
# #SBATCH -t 00:30:00
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --output=/cfs/klemming/scratch/p/penotcam/mlls/new_test/notion_postprocess_%j.log
# #SBATCH --error=/cfs/klemming/scratch/p/penotcam/mlls/new_test/notion_postprocess_%j.err
# set -euo pipefail
# cd /cfs/klemming/scratch/p/penotcam/mlls/new_test
# export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# srun ../.aspire-env/bin/python -u ../test_particle_segmentation.py
# srun ../.aspire-env/bin/python -u notion_postprocess.py --output-dir "notion_postprocess_${SLURM_JOB_ID}" "$@"

# --- Original guide block 17 ---
# #!/bin/bash
# #SBATCH -A naiss2025-3-71
# #SBATCH -J halo_postprocess
# #SBATCH -p main
# #SBATCH -t 00:30:00
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --output=/cfs/klemming/scratch/p/penotcam/mlls/new_test/halo_postprocess_%j.log
# #SBATCH --error=/cfs/klemming/scratch/p/penotcam/mlls/new_test/halo_postprocess_%j.err
# set -euo pipefail
# cd /cfs/klemming/scratch/p/penotcam/mlls/new_test
# export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# srun ../.aspire-env/bin/python -u ../test_particle_segmentation.py
# srun ../.aspire-env/bin/python -u test_halo_segmentation.py
# srun ../.aspire-env/bin/python -u pilot_halo.py
# srun ../.aspire-env/bin/python -u notion_postprocess.py --ignore-halo --output-dir "halo_postprocess_${SLURM_JOB_ID}" "$@"

