# =======================
## SELF-PRETRAINED MODEL
# =======================

#  ------ original order ------

# no reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu"

# coconut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut"

# multimodal coconut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut"

# modified multinut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut"

# ------ swapped order ------

# no reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu"

# # coconut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut"

# multimodal coconut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut"

# modified multinut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut"
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut"

# ==========================
## PROVIDED PRETRAINED MODEL
# ==========================

# ------ original order ------

# no reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt

# coconut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt

# multimodal coconut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt

# modified multinut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --modified-multinut --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/normal-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt

# ------ swapped order ------

# no reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-pretrained-ckpt-dir" --pretrained-ckpt

# coconut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-coconut-with-pretrained-ckpt-dir" --pretrained-ckpt

# multimodal coconut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt

# modified multinut reasoning
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset scienceqa --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmmu --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
torchrun --nproc_per_node 1 evaluate.py --cfg-path eval_configs/evaluate_image.yaml --eval-dataset image_val --output-dir self-trained --dataset mmstar --coconut --modified-multinut --swap-order --ckpt-dir "/home/csalt/Haider/DVLM/OmniMod/OmniMod/swapped-finetuning-mmmu-with-modified-multinut-with-pretrained-ckpt-dir" --pretrained-ckpt
