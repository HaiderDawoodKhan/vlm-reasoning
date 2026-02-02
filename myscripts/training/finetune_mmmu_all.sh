# baseline
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-coconut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-coconut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-multinut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-multinut"


# swapped
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-coconut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-coconut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-multinut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-multinut"


# original with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-modified-multinut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-modified-multinut"


# swapped with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-modified-multinut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --swap-order --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-modified-multinut"

# pretrained (pre-training) checkpoint
# ====================================
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-pretrained-ckpt"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-coconut-with-pretrained-ckpt"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-multinut-with-pretrained-ckpt"


# swapped
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-pretrained-ckpt"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-coconut-with-pretrained-ckpt"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-multinut-with-pretrained-ckpt"


# original with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-modified-multinut-with-pretrained-ckpt"


# swapped with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --swap-order --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-modified-multinut-with-pretrained-ckpt"