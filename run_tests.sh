# python scripts/molnet_test_runner.py delaney gnn --splitter scaffold --n 10 --max-epochs 100 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py delaney srgnn --splitter scaffold --n 10 --max-epochs 100 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges

# python scripts/molnet_test_runner.py lipo gnn --splitter scaffold --n 10 --max-epochs 100 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py lipo srgnn --splitter scaffold --n 10 --max-epochs 100 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges

# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --splitter scaffold --n 10 --max-epochs 100 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --splitter scaffold --n 10 --max-epochs 100 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges

# python scripts/molnet_test_runner.py freesolv gnn --n 10 --max-epochs 100 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py freesolv srgnn --n 10 --max-epochs 100 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges

# python scripts/molnet_test_runner.py lipo gnn --n 10 --max-epochs 100 --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py lipo srgnn --n 10 --max-epochs 100 --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges

# python scripts/molnet_test_runner.py qm8 gnn --n 1 --max-epochs 100 --monitor mae --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py qm8 srgnn --n 1 --max-epochs 100 --monitor mae --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 128 --n-hidden-channels 32

# python scripts/molnet_test_runner.py ocelot gnn --splitter scaffold --n 1 --max-epochs 100 --monitor mae --project gine-baselines --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges
# python scripts/molnet_test_runner.py ocelot srgnn --splitter scaffold --n 1 --max-epochs 100 --monitor mae --project gine-baselines --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32 --no-charges --variant no_charges

# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges

python scripts/molnet_test_runner.py delaney msr1 --n 10 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges
python scripts/molnet_test_runner.py delaney msr2 --n 10 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges

python scripts/molnet_test_runner.py lipo msr1 --n 10 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges
python scripts/molnet_test_runner.py lipo msr2 --n 10 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges

python scripts/molnet_test_runner.py delaney msr1 --n 10 --max-epochs 150 --splitter scaffold --project gine-baselines --no-charges --variant no_charges
python scripts/molnet_test_runner.py delaney msr2 --n 10 --max-epochs 150 --splitter scaffold --project gine-baselines --no-charges --variant no_charges

python scripts/molnet_test_runner.py lipo msr1 --n 10 --max-epochs 150 --splitter scaffold --project gine-baselines --no-charges --variant no_charges
python scripts/molnet_test_runner.py lipo msr2 --n 10 --max-epochs 150 --splitter scaffold --project gine-baselines --no-charges --variant no_charges

python scripts/molnet_test_runner.py clintox msr1 --task-type classification --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges
python scripts/molnet_test_runner.py clintox msr2 --task-type classification --splitter scaffold --n 10 --max-epochs 150 --project gine-baselines --no-charges --variant no_charges

python scripts/molnet_test_runner.py freesolv msr1 --n 10 --max-epochs 150 --splitter scaffold --project gine-baselines --no-charges --variant no_charges
python scripts/molnet_test_runner.py freesolv msr2 --n 10 --max-epochs 150 --splitter scaffold --project gine-baselines --no-charges --variant no_charges

# python scripts/molnet_test_runner.py doyle msr2 --n 5 --max-epochs 200 --n-hidden-sets 128 --n-elements 128 --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 64 --n-hidden-channels 32