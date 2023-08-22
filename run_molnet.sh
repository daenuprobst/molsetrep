############
# Hyperparameter tuning on GNN
############
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 64 --n-hidden-channels 32 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --n-layers 4 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --n-layers 8 --variant hyper-tuning
python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 4 --variant hyper-tuning
python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning


# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 64 --n-hidden-channels 32 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --n-layers 4 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --n-layers 8 --variant hyper-tuning
python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 4 --variant hyper-tuning
python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning











# python scripts/molnet_test_runner.py lipo gnn --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py lipo srgnn --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py lipo msr1  --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --no-charges --variant test --n-hidden-sets 64 --n-hidden-sets 64
# python scripts/molnet_test_runner.py lipo msr2  --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --no-charges --variant test --n-hidden-sets 64 --n-hidden-sets 64

# python scripts/molnet_test_runner.py freesolv gnn --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py freesolv srgnn --monitor loss --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py freesolv msr1 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --no-charges --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py freesolv msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --project gine-baselines --n-hidden-sets 64 --n-hidden-sets 64 --no-charges --variant no_charges
