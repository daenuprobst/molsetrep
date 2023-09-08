############
# Doyle
############
# python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 450 --project rxn-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.7
# python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 450 --project rxn-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.5
# python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 450 --project rxn-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.3
# python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 450 --project rxn-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.2
# python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 450 --project rxn-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.1
# python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 450 --project rxn-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.05
# python scripts/molnet_test_runner.py doyle msr2 --monitor loss --splitter scaffold --n 10 --max-epochs 450 --project rxn-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 16 --split-ratio 0.025

############
# Doyle Test
############
python scripts/molnet_test_runner.py doyle_test msr2 --monitor loss --splitter scaffold --n 12 --max-epochs 150 --project rxn-final --n-hidden-sets 32 --n-hidden-sets 32 --n-elements 4 --n-elements 4 --batch-size 16 --learning-rate 0.01 --variant ttfp-32-4

############
# AZ
############
# python scripts/molnet_test_runner.py az srgnn --monitor loss --splitter scaffold --n 10 --max-epochs 450 --project rxn-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --batch-size 64 --split-ratio 0.7 --variant transformer --set-layer transformer
