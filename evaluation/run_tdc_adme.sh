############
# TDC ADME
############

# Spearman stuff
# python ../scripts/tdc_test_runner.py --max-epochs 500 --gnn-dropout 0.0 --learning-rate 0.001 --n-elements 128 --n-hidden-sets 128 --no-pool --variant combinedloss_longer_nopool
python ../scripts/tdc_test_runner.py --max-epochs 350 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 128 --n-hidden-sets 128
python ../scripts/tdc_test_runner.py --max-epochs 350 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 64 --n-hidden-sets 64
python ../scripts/tdc_test_runner.py --max-epochs 350 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 32 --n-hidden-sets 32
python ../scripts/tdc_test_runner.py --max-epochs 350 --gnn-dropout 0.1 --learning-rate 0.001 --n-elements 8 --n-hidden-sets 8
