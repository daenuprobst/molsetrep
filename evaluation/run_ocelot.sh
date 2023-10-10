############
# OCELOT
############
../scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128
../scripts/molnet_test_runner.py ocelot gnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128
../scripts/molnet_test_runner.py ocelot msr1 --monitor loss --splitter scaffold --n 5 --max-epochs 250 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128
../scripts/molnet_test_runner.py ocelot msr2 --monitor loss --splitter scaffold --n 5 --max-epochs 250 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128