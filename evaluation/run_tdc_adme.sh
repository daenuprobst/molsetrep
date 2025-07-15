############
# TDC ADME
############

# python ../scripts/tdc_test_runner.py --max-epochs 350 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 32 --n-hidden-sets 32 --no-pool --variant "log scaled"

# Spearman stuff
python ../scripts/tdc_test_runner.py --max-epochs 350 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 16 --n-hidden-sets 16 --n-layers 2 --variant "simple"
