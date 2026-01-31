
# original order
# no reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset scienceqa
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmmu
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmstar

# # coconut reasoning
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset scienceqa --coconut
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmmu --coconut
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmstar --coconut


# # multimodal coconut reasoning
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset scienceqa --coconut --multinut
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmmu --coconut --multinut
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmstar --coconut --multinut



# swapped order
# no reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset scienceqa --swap-order
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmmu --swap-order
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmstar --swap-order

# # coconut reasoning
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset scienceqa --coconut --swap-order
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmmu --coconut --swap-order
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmstar --coconut --swap-order


# # multimodal coconut reasoning
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset scienceqa --coconut --multinut --swap-order
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmmu --coconut --multinut --swap-order
# torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir pretrained --dataset mmstar --coconut --multinut --swap-order