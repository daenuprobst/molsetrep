############
# TDC ADME
############
python ../scripts/molnet_test_runner.py tdc_adme srgnn --monitor loss --splitter scaffold --n 3 --max-epochs 400 --n-hidden-sets 64 --n-elements 4 --n-hidden-channels 256 --n-hidden-channels 128 --n-layers 8 --set-layer transformer
# python ../scripts/molnet_test_runner.py tdc_adme gnn --monitor loss --splitter scaffold --n 3 --max-epochs 450 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python ../scripts/molnet_test_runner.py tdc_adme msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --n-hidden-sets 64 --n-elements 4
# python ../scripts/molnet_test_runner.py tdc_adme msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4
