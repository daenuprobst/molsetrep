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


# python scripts/molnet_test_runner.py ocelot msr2 --monitor loss --n 1 --max-epochs 250 --project gine-baselines --n-hidden-sets 64 --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant test
python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant tuned