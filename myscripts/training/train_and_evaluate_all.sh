# ====================================
# pretrained (pre-training) checkpoint
# ====================================

# baseline
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-pretrained-ckpt"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt

# coconut reasoning
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-coconut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-coconut-with-pretrained-ckpt"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt

# multimodal coconut reasoning
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-multinut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --multinut --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-multinut-with-pretrained-ckpt"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt

# swapped baseline
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --swap-order --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-pretrained-ckpt"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt

# swapped coconut
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-coconut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --swap-order --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-coconut-with-pretrained-ckpt"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt

# swapped multimodal coconut reasoning
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-multinut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --multinut --swap-order --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-multinut-with-pretrained-ckpt"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt

# original with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-modified-multinut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --modified_multinut --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-modified-multinut-with-pretrained-ckpt"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt

# swapped with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-modified-multinut-with-pretrained-ckpt --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --swap-order --coconut --modified_multinut --ckpt "/home/csalt/Haider/DVLM/Trained-Checkpoints/VQAv2_checkpoint_1.pth"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --swap-order --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-modified-multinut-with-pretrained-ckpt"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt

# ====================================
# self pretraining checkpoint
# ====================================

# baseline
# torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir normal-pretraining --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-pretraining"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu"

# coconut
# torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir normal-pretraining-with-coconut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-coconut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-pretraining-with-coconut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-coconut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-coconut"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut"

# multinut
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir normal-pretraining-with-multinut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut --multinut
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-multinut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-pretraining-with-multinut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-multinut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-multinut"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut"


# swapped baseline
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir swapped-pretraining --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --swap-order
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-pretraining"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu"

# swapped coconut
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir swapped-pretraining-with-coconut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut --swap-order
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-coconut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-pretraining-with-coconut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-coconut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-coconut"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut"

# swapped multimodal coconut reasoning
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir swapped-pretraining-with-multinut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut --multinut --swap-order
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-multinut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-pretraining-with-multinut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-multinut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-multinut"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut"

# original with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir normal-pretraining-with-modified-multinut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut --modified_multinut
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir normal-finetuning-scienceqa-with-modified-multinut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-pretraining-with-modified-multinut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir normal-finetuning-mmmu-with-modified-multinut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-scienceqa-with-modified-multinut"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut"

# swapped with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir swapped-pretraining-with-modified-multinut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --swap-order --coconut --modified_multinut
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset scienceqa --output-dir swapped-finetuning-scienceqa-with-modified-multinut --max-epoch 10 --warmup-steps 200 --iters-per-epoch 2073 --swap-order --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-pretraining-with-modified-multinut"
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset mmmu --output-dir swapped-finetuning-mmmu-with-modified-multinut --max-epoch 10 --warmup-steps 5 --iters-per-epoch 50 --swap-order --coconut --modified_multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-scienceqa-with-modified-multinut"

torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut"

