# baseline
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-pretraining"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-coconut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-pretraining-with-coconut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-multinut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-pretraining-with-multinut"


# swapped
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-pretraining"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-coconut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-pretraining-with-coconut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-multinut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-pretraining-with-multinut"


# original with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-modified-multinut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-pretraining-with-modified-multinut"


# swapped with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-modified-multinut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --swap-order --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-pretraining-with-modified-multinut"

# pretrained (pre-training) checkpoint
# ====================================
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-coconut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-multinut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --multinut --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"


# swapped
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --swap-order --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-coconut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --swap-order --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-multinut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --multinut --swap-order --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"


# original with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-modified-multinut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --modified_multinut --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"


# swapped with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-modified-multinut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --swap-order --coconut --modified_multinut --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"