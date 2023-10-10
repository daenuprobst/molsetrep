############
# Doyle
############
python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.025 --learning-rate 0.01
python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.05 --learning-rate 0.01
python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.1 --learning-rate 0.01
python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.2 --learning-rate 0.01
python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.3 --learning-rate 0.01
python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.5 --learning-rate 0.01
python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.7 --learning-rate 0.01

############
# Doyle Test
############
python scripts/molnet_test_runner.py doyle_test msr2 --monitor loss --splitter scaffold --n 16 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --learning-rate 0.01

############
# AZ
############
python scripts/molnet_test_runner.py az msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 250 --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.7 --learning-rate 0.01
