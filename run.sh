export CUDA_VISIBLE_DEVICES=1

python main.py --model-name huggyllama/llama-7b --dataset_type multiple_choice --mode DOLA --layer 33 --name DOLA
