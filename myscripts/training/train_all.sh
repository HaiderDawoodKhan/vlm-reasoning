# baseline
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir normal-pretraining --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir normal-pretraining-with-coconut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir normal-pretraining-with-multinut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut --multinut


# swapped
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir swapped-pretraining --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --swap-order
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir swapped-pretraining-with-coconut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut --swap-order
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir swapped-pretraining-with-multinut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut --multinut --swap-order


# original with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir normal-pretraining-with-modified-multinut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --coconut --modified_multinut


# swapped with modified attention
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/train_image.yaml --eval-dataset image_val --dataset vqav2 --output-dir swapped-pretraining-with-modified-multinut --max-epoch 1 --warmup-steps 57688 --iters-per-epoch 147919 --swap-order --coconut --modified_multinut
