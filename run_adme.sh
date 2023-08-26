############
# ADME
############
python scripts/molnet_test_runner.py adme srgnn  --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task HLM
python scripts/molnet_test_runner.py adme gnn --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task HLM
python scripts/molnet_test_runner.py adme msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4 --task HLM
python scripts/molnet_test_runner.py adme msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task HLM

python scripts/molnet_test_runner.py adme srgnn  --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task MDR1_ER
python scripts/molnet_test_runner.py adme gnn --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task MDR1_ER
python scripts/molnet_test_runner.py adme msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4 --task MDR1_ER
python scripts/molnet_test_runner.py adme msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task MDR1_ER

python scripts/molnet_test_runner.py adme srgnn  --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task Sol
python scripts/molnet_test_runner.py adme gnn --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task Sol
python scripts/molnet_test_runner.py adme msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4 --task Sol
python scripts/molnet_test_runner.py adme msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task Sol

python scripts/molnet_test_runner.py adme srgnn  --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task RLM
python scripts/molnet_test_runner.py adme gnn --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task RLM
python scripts/molnet_test_runner.py adme msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4 --task RLM
python scripts/molnet_test_runner.py adme msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task RLM

python scripts/molnet_test_runner.py adme srgnn  --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task hPPB
python scripts/molnet_test_runner.py adme gnn --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task hPPB
python scripts/molnet_test_runner.py adme msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4 --task hPPB
python scripts/molnet_test_runner.py adme msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task hPPB

python scripts/molnet_test_runner.py adme srgnn  --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task rPPB
python scripts/molnet_test_runner.py adme gnn --monitor loss --splitter scaffold --n 3 --max-epochs 900 --project moleculenet-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task rPPB
python scripts/molnet_test_runner.py adme msr1 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-elements 4 --task rPPB
python scripts/molnet_test_runner.py adme msr2 --monitor loss --splitter scaffold --n 3 --max-epochs 250 --project moleculenet-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task rPPB