export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/ubuntu/input_data/banana_boat"
export OUTPUT_DIR="/home/ubuntu/0417_model_banana_boat_v2/"
  
python3 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of S*A*" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --seed=19960520 \