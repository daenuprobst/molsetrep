############
# ADME
############
python scripts/molnet_test_runner.py ocelot srgnn  --monitor loss --splitter scaffold --n 5 --max-epochs 150 --project ocelot-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task homo
python scripts/molnet_test_runner.py ocelot gnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --project ocelot-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task homo
python scripts/molnet_test_runner.py ocelot msr1 --monitor loss --splitter scaffold --n 5 --max-epochs 250 --project ocelot-final --n-hidden-sets 64 --n-elements 4 --task homo
python scripts/molnet_test_runner.py ocelot msr2 --monitor loss --splitter scaffold --n 5 --max-epochs 250 --project ocelot-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task homo