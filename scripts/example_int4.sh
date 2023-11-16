MODEL_PATH=$1  # The path to the checkpoint
IMAGE_FILE=$2  # The path to a jpg/png

CUDA_VISIBLE_DEVICES=0 python -m examples.example_chat --model-path $MODEL_PATH \
                                                       --image-file $IMAGE_FILE \
                                                       --load-4bit