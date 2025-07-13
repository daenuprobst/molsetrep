############
# TDC ADME
############

# python ../scripts/tdc_test_runner.py --max-epochs 350 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 32 --n-hidden-sets 32 --no-pool --variant "log scaled"

# Spearman stuff
# python ../scripts/tdc_test_runner.py --max-epochs 200 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 128 --n-hidden-sets 128 --n-layers 1
# python ../scripts/tdc_test_runner.py --max-epochs 200 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 64 --n-hidden-sets 64 --n-layers 1
python ../scripts/tdc_test_runner.py --max-epochs 350 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 32 --n-hidden-sets 32 --n-layers 5 --variant "nomlp_spearman_only_l2_10"

# TODO: Try regressor without MLP at the end
