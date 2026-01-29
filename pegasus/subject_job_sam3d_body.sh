#!/bin/bash

# 設定: 合計人数と1ジョブあたりの人数
TOTAL_PERSONS=22
PERSONS_PER_JOB=2
NUM_JOBS=$((TOTAL_PERSONS / PERSONS_PER_JOB))

# 実行するシェルスクリプト名
TARGET_SCRIPT="run_sam3d_body.sh"

for ((i=0; i<NUM_JOBS; i++))
do
    # 担当するPerson IDの範囲を計算
    START_ID=$((i * PERSONS_PER_JOB))
    END_ID=$((START_ID + 1))
    
    # Hydraに渡すためのリスト形式を作成 "[0,1]"
    PERSON_LIST="[$START_ID,$END_ID]"
    
    # ジョブ名を識別しやすく設定 (例: sam3d_p0_1)
    JOB_NAME="sam3d_p${START_ID}_${END_ID}"
    
    echo "Submitting Job $((i+1))/$NUM_JOBS: $JOB_NAME (Persons: $PERSON_LIST)"
    
    # qsub の実行
    # -v: 環境変数 (PERSON_LIST, JOB_ID) を実行スクリプトへ渡す
    # -N: ジョブ名を指定
    qsub -N "$JOB_NAME" \
         -v "PERSON_LIST=$PERSON_LIST,JOB_ID=$i" \
         "$TARGET_SCRIPT"
done

echo "✅ All 11 jobs have been submitted."