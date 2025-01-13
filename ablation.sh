export CUDA_VISIBLE_DEVICES=7

python ablation.py --dataset_name "SPC" --factor "PTE" --no_E
python ablation.py --dataset_name "SPC" --factor "PTE" --no_T
python ablation.py --dataset_name "SPC" --factor "PTE" --no_P
python ablation.py --dataset_name "SPC" --factor "PTE" --no_E --no_T
python ablation.py --dataset_name "SPC" --factor "PTE" --no_E --no_T --no_P

python ablation.py --dataset_name "PC" --factor "PTE" --no_E
python ablation.py --dataset_name "PC" --factor "PTE" --no_T
python ablation.py --dataset_name "PC" --factor "PTE" --no_P
python ablation.py --dataset_name "PC" --factor "PTE" --no_E --no_T
python ablation.py --dataset_name "PC" --factor "PTE" --no_E --no_T --no_P
