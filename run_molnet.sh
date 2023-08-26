
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 16
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 32
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 64


# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 16  --n-hidden-sets 16
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 32 --n-hidden-sets 32
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 64 --n-hidden-sets 64

# python scripts/molnet_test_runner.py clintox msr1 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 350 --project moleculenet-final --n-hidden-sets 64 --n-elements 4


# python scripts/molnet_test_runner.py delaney msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py delaney msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

# python scripts/molnet_test_runner.py lipo msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py lipo msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

# python scripts/molnet_test_runner.py delaney gnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py delaney srgnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8

# python scripts/molnet_test_runner.py lipo gnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8

# python scripts/molnet_test_runner.py hiv msr1 --task-type classification --monitor loss --splitter scaffold --n 1 --start-n 2 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py hiv gnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8

python scripts/molnet_test_runner.py hiv srgnn --task-type classification --monitor loss --splitter scaffold --n 1 --start-n 2 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
python scripts/molnet_test_runner.py hiv msr2 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4
