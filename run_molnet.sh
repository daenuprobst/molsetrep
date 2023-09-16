
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 16
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 32
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 64

############
# Hyperparameter tuning on GNN
############

## GINE
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 64 --n-hidden-channels 32 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --n-layers 4 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 4 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning

## SR-GINE
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 64 --n-hidden-channels 32 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --n-layers 4 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 32 --n-hidden-channels 16 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 4 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning

# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 16  --n-hidden-sets 16
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 32 --n-hidden-sets 32
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --no-charges --n-hidden-sets 64 --n-hidden-sets 64

# python scripts/molnet_test_runner.py clintox msr1 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 350 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
## Set layer hyper params for SR-GINE
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 16 --n-elements 16 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 16 --n-elements 8 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 32 --n-elements 8 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 32 --n-elements 16 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 64 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 64 --n-elements 32 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 128 --n-elements 128 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 150 --project bbbp-hyperparam --n-hidden-sets 128 --n-elements 32 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --variant hyper-tuning

## MSR1
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 32 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 64 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 64 --n-elements 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 64 --n-elements 4 --variant hyper-tuning

# python scripts/molnet_test_runner.py delaney msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py delaney msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

# python scripts/molnet_test_runner.py lipo msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py lipo msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

# python scripts/molnet_test_runner.py delaney gnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py delaney srgnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8

# python scripts/molnet_test_runner.py lipo gnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
## MSR2
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 16 --n-hidden-sets 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 32 --n-hidden-sets 32 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 64 --n-hidden-sets 64 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 16 --n-elements 16 --variant hyper-tuning
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 6 --max-epochs 250 --project bbbp-hyperparam --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --variant hyper-tuning

############
# BACE
############
# python scripts/molnet_test_runner.py bace_classification gnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py bace_classification srgnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py bace_classification msr1 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py bace_classification msr2 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

# python scripts/molnet_test_runner.py bace msr1 --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py bace msr2 --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4
# python scripts/molnet_test_runner.py bace gnn --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py bace srgnn --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8

############
# HIV
############
# python scripts/molnet_test_runner.py hiv msr1 --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py hiv msr2 --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4
# python scripts/molnet_test_runner.py hiv gnn --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py hiv srgnn --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8


############
# ClinTox
############
# python scripts/molnet_test_runner.py clintox gnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py clintox srgnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py clintox msr1 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py clintox msr2 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

############
# Sider
############
# python scripts/molnet_test_runner.py sider gnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py sider srgnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py sider msr1 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py sider msr2 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

############
# BBBP
############
# python scripts/molnet_test_runner.py bbbp gnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py bbbp srgnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py bbbp msr1 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py bbbp msr2 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

############
# ClinTox
############
# python scripts/molnet_test_runner.py freesolv gnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py freesolv srgnn --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py freesolv msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py freesolv msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

# python scripts/molnet_test_runner.py hiv msr1 --task-type classification --monitor loss --splitter scaffold --n 1 --start-n 2 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py hiv gnn --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 150 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py hiv srgnn --task-type classification --monitor loss --splitter scaffold --n 1 --start-n 2 --max-epochs 150 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py hiv msr2 --task-type classification --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4

############
# Tox21
############
# python scripts/molnet_test_runner.py tox21 msr1 --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py tox21 msr2 --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4
# python scripts/molnet_test_runner.py tox21 gnn --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py tox21 srgnn --task-type classification --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8

############
# QM7
############
# python scripts/molnet_test_runner.py qm7 msr1 --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py qm7 msr2 --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4
# python scripts/molnet_test_runner.py qm7 gnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py qm7 srgnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8

############
# QM8
############
# python scripts/molnet_test_runner.py qm8 msr1 --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py qm8 msr2 --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4
python scripts/molnet_test_runner.py qm8 gnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8  --task E1-CAM --task E2-CAM --task f1-CAM --task f2-CAM
# python scripts/molnet_test_runner.py qm8 srgnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8


############
# QM9
############
# python scripts/molnet_test_runner.py qm9 msr1 --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-elements 4
# python scripts/molnet_test_runner.py qm7 msr2 --monitor loss --splitter custom-scaffold --n 3 --max-epochs 250 --project moleculenet-custom-splitter --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4
# python scripts/molnet_test_runner.py qm7 gnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
# python scripts/molnet_test_runner.py qm7 srgnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project moleculenet-custom-splitter --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8
