python main.py \
    --tpu=$1 \
    --mode='train' \
    --data_dir=$2 \
    --model_dir=$3 \
    --model_name='efficientnet-b3' \
    --skip_host_call=true \
    --train_batch_size=1024 \
    --train_steps=500000 \
    --base_learning_rate=0.005
