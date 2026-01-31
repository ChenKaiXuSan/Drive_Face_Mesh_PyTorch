#!/bin/bash
#PBS -A SKIING
#PBS -q gen_S
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_person_20_only
#PBS -o logs/pegasus/sam3d_p20.log
#PBS -e logs/pegasus/sam3d_p20_err.log

# === 1. 環境準備 ===
cd /work/SKIING/chenkaixu/code/Drive_Face_Mesh_PyTorch
mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body

# === 2. パス設定 ===
VIDEO_PATH="/work/SKIING/chenkaixu/data/drive/videos_split"
RESULT_PATH="/work/SKIING/chenkaixu/data/drive/sam3d_body_results_2"
START_MID_END_PATH="/work/SKIING/chenkaixu/data/drive/annotation/split_mid_end/mini.json"
CKPT_ROOT="/work/SKIING/chenkaixu/code/Drive_Face_Mesh_PyTorch/ckpt/sam-3d-body-dinov3"

# === 3. 実行パラメータ ===
# 20号のみを指定
PERSON_LIST="[20]"

echo "🚀 Starting processing for Person: 20"
echo "Started at: $(date)"

# === 4. 実行 ===
# 1人（7環境）なので workers_per_gpu=7 に設定
python -m SAM3Dbody.main \
    paths.video_path=${VIDEO_PATH} \
    paths.result_output_path=${RESULT_PATH} \
    model.root_path=${CKPT_ROOT} \
    infer.gpu="[0]" \
    infer.workers_per_gpu=7 \
    infer.person_list="${PERSON_LIST}" \
    paths.start_mid_end_path=${START_MID_END_PATH}

echo "✅ Finished Person 20 at: $(date)"