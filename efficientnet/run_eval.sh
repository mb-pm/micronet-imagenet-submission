python main.py \
    --use_tpu=False \
    --data_dir=$1 \
    --model_dir=$2 \
    --model_name='efficientnet-b3' \
    --mode='eval' \
    --eval_batch_size=100
