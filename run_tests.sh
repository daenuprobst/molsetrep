# python scripts/molnet_test_runner.py lipo gnn --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py lipo srgnn --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py lipo msr1  --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --no-charges --variant test --n-hidden-sets 64 --n-hidden-sets 64
# python scripts/molnet_test_runner.py lipo msr2  --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --no-charges --variant test --n-hidden-sets 64 --n-hidden-sets 64

# python scripts/molnet_test_runner.py freesolv gnn --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py freesolv srgnn --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py freesolv msr1 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --no-charges --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py freesolv msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant no_charges

# python scripts/molnet_test_runner.py clintox gnn --task-type classification --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py clintox srgnn --task-type classification --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py clintox msr1 --task-type classification --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --no-charges --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py clintox msr2 --task-type classification --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant no_charges

# python scripts/molnet_test_runner.py hiv gnn --task-type classification --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py hiv srgnn --task-type classification --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py hiv msr1 --task-type classification --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --no-charges --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py hiv msr2 --task-type classification --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant no_charges







# python scripts/molnet_test_runner.py ocelot srgnn --splitter scaffold --n 2 --max-epochs 150 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py ocelot gnn --splitter scaffold --n 2 --max-epochs 150 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py ocelot msr1 --splitter scaffold --n 2 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges --n-hidden-sets 64 --n-hidden-sets 64
# python scripts/molnet_test_runner.py ocelot msr2 --splitter scaffold --n 2 --max-epochs 250 --project gine-baselines --no-charges --variant no_charges --n-hidden-sets 64 --n-hidden-sets 64


# python scripts/molnet_test_runner.py doyle msr2 --splitter scaffold --n 2 --max-epochs 250 --project gine-baselines --n-hidden-sets 32 --n-hidden-sets 32 --n-elements 16 --n-elements 16 --n-hidden-channels 128 --n-hidden-channels 32 --no-charges --variant fp-2048 --split-ratio 0.5
# python scripts/molnet_test_runner.py doyle msr2 --splitter scaffold --n 2 --max-epochs 250 --project gine-baselines --n-hidden-sets 32 --n-hidden-sets 32 --n-elements 16 --n-elements 16 --n-hidden-channels 128 --n-hidden-channels 32 --no-charges --variant fp-2048 --split-ratio 0.3
# python scripts/molnet_test_runner.py doyle msr2 --splitter scaffold --n 2 --max-epochs 250 --project gine-baselines --n-hidden-sets 32 --n-hidden-sets 32 --n-elements 16 --n-elements 16 --n-hidden-channels 128 --n-hidden-channels 32 --no-charges --variant fp-2048 --split-ratio 0.2
# python scripts/molnet_test_runner.py doyle msr2 --splitter scaffold --n 2 --max-epochs 250 --project gine-baselines --n-hidden-sets 32 --n-hidden-sets 32 --n-elements 16 --n-elements 16 --n-hidden-channels 128 --n-hidden-channels 32 --no-charges --variant fp-2048 --split-ratio 0.1
# python scripts/molnet_test_runner.py doyle msr2 --splitter scaffold --n 2 --max-epochs 250 --project gine-baselines --n-hidden-sets 32 --n-hidden-sets 32 --n-elements 16 --n-elements 16 --n-hidden-channels 128 --n-hidden-channels 32 --no-charges --variant fp-2048 --split-ratio 0.05
# python scripts/molnet_test_runner.py doyle msr2 --splitter scaffold --n 2 --max-epochs 250 --project gine-baselines --n-hidden-sets 32 --n-hidden-sets 32 --n-elements 16 --n-elements 16 --n-hidden-channels 128 --n-hidden-channels 32 --no-charges --variant fp-2048 --split-ratio 0.025

# python scripts/molnet_test_runner.py uspto msr2 --splitter scaffold --n 2 --max-epochs 250 --project gine-baselines --no-charges --variant fp-2048 --split-ratio 0.7

# python scripts/molnet_test_runner.py suzuki msr2 --splitter scaffold --n 1 --max-epochs 250 --project gine-baselines --split-ratio 0.7 --max-epochs 50 --learning-rate 0.0001

# python scripts/molnet_test_runner.py adme msr2 --monitor loss --n 2 --max-epochs 200 --project gine-baselines
# python scripts/molnet_test_runner.py adme msr1 --monitor loss --n 2 --max-epochs 200 --project gine-baselines
# python scripts/molnet_test_runner.py adme srgnn --monitor loss --n 2 --max-epochs 600 --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --project gine-baselines

python scripts/molnet_test_runner.py adme msr2 --monitor loss --splitter scaffold --n 2 --max-epochs 600 --project gine-baselines --n-hidden-sets 64 --n-hidden-sets 64
