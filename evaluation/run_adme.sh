############
# ADME
############
python ../scripts/molnet_test_runner.py adme srgnn  --monitor loss --splitter scaffold --n 3 --max-epochs 900 --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
python ../scripts/molnet_test_runner.py adme gnn --monitor loss --splitter scaffold --n 3 --max-epochs 900 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
python ../scripts/molnet_test_runner.py adme msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --n-hidden-sets 64 --n-elements 4
python ../scripts/molnet_test_runner.py adme msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4