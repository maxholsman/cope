export CUDA_VISIBLE_DEVICES=1

python -u generate.py \
--config /scratch/pranamlab/tong/cope/editflows/configs/config_test.yaml \
--ckpt /scratch/pranamlab/tong/cope/editflows/outputs/2025.11.05/011146/checkpoint/epoch0090-val29.90.ckpt \
--input 'CC(C)[C@H](NC(=O)[C@H](C)NC(=O)[C@H](C)N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@H](C(=O)N[C@@H](C)C(=O)O)C(C)C'
