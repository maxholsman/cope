export CUDA_VISIBLE_DEVICES=1

python -u generate.py \
--config /scratch/pranamlab/tong/cope/editflows/configs/config_test.yaml \
--ckpt /scratch/pranamlab/tong/cope/editflows/outputs/2025.11.05/011146/checkpoint/epoch0090-val29.90.ckpt \
--op_temperature 1 \
--token_temperature 1 \
--input 'CC[C@H](C)[C@@H](C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](CCSC)C(=O)N(C)[C@H](C(=O)N(C)CC(N)=O)C(C)C)N(C)C(C)=O'
