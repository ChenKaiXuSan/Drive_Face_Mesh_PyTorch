#!/bin/bash
#PBS -A SKIING
#PBS -q gen_S
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_4nodes_run
#PBS -t 0-21                           # 22个
#PBS -o logs/pegasus/sam3d_group_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/sam3d_group_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
cd /work/SKIING/chenkaixu/code/Drive_Face_Mesh_PyTorch

mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body

# --- マッピング定義 (Dictionary形式) ---
# 11個のサブジョブ (0-10) に対応するマッピング
declare -A ID_MAP

# 0から10までのキーにそれぞれリストを割り当て
ID_MAP["0"]="[01]"
ID_MAP["1"]="[02]"
ID_MAP["2"]="[03]"
ID_MAP["3"]="[04]"
ID_MAP["4"]="[05]"
ID_MAP["5"]="[06]"
ID_MAP["6"]="[07]"
ID_MAP["7"]="[08]"
ID_MAP["8"]="[09]"
ID_MAP["9"]="[10]"
ID_MAP["10"]="[11]"
ID_MAP["11"]="[12]"
ID_MAP["12"]="[13]"
ID_MAP["13"]="[14]"
ID_MAP["14"]="[15]"
ID_MAP["15"]="[16]"
ID_MAP["16"]="[17]"
ID_MAP["17"]="[18]"
# ID_MAP["18"]="[19]" # down ws2
ID_MAP["19"]="[20]"
# ID_MAP["20"]="[21]" # down ccs
# ID_MAP["21"]="[24]" # down ws2

# 現在のタスク用リストを取得 (PBS_SUBREQNO は 0-21 の値をとる想定)
PERSON_LIST=${ID_MAP[$PBS_SUBREQNO]}

echo "Node Index: $PBS_SUBREQNO"
echo "Processing folders: $PERSON_LIST"

# === 3. パス設定と実行 ===
VIDEO_PATH="/work/SKIING/chenkaixu/data/drive/videos_split"
RESULT_PATH="/work/SKIING/chenkaixu/data/drive/sam3d_body_results"
CKPT_ROOT="/work/SKIING/chenkaixu/code/Drive_Face_Mesh_PyTorch/ckpt/sam-3d-body-dinov3"

python -m SAM3Dbody.main \
    paths.video_path=${VIDEO_PATH} \
    paths.result_output_path=${RESULT_PATH} \
    model.root_path=${CKPT_ROOT} \
    infer.gpu="[0]" \
    infer.workers_per_gpu=7 \
    infer.person_list="${PERSON_LIST}" \

echo "🏁 Node ${PBS_SUBREQNO} finished at: $(date)"
# 一个node里面跑一个人的4个环境，也就是4个worker