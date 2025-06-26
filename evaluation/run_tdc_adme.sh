############
# TDC ADME
############
# python ../scripts/molnet_test_runner.py tdc_adme srgnn --monitor mae --splitter scaffold --n 3 --max-epochs 450 --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 168 --n-hidden-channels 200 --n-layers 8 --gnn-dropout 0.0 --learning-rate 0.001 --variant "test_lip"

# Bioavail
# python ../scripts/molnet_test_runner.py tdc_adme srgnn --task-type classification --monitor auroc --splitter scaffold --n 3 --max-epochs 450 --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 168 --n-hidden-channels 200 --n-layers 8 --gnn-dropout 0.5 --learning-rate 0.0001 --variant "bioavail"

# CYP3A4_Substrate_CarbonMangels
python ../scripts/molnet_test_runner.py tdc_adme srgnn --task-type classification --monitor auroc --splitter scaffold --n 3 --max-epochs 100 --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 64 --n-hidden-channels 128 --n-layers 8 --gnn-dropout 0.1 --learning-rate 0.0001 --use-class-weights --variant "cyp_class_weights_dropout0.1_lr0.0001_64_128_longer"
