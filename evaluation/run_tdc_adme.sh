############
# TDC ADME
############
# python ../scripts/molnet_test_runner.py tdc_adme srgnn --monitor mae --splitter scaffold --n 3 --max-epochs 400 --learning-rate 0.001 --n-layers 12
python ../scripts/molnet_test_runner.py tdc_adme srgnn --monitor mae --splitter scaffold --n 3 --max-epochs 400 --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --gnn-dropout 0.0 --learning-rate 0.001 --variant "jklstm_global_gine_dropout0.0_gat_lip_leakyrelu"
