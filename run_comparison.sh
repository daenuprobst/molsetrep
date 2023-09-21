# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer setrep
# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer transformer
# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer deepset

# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer setrep --gnn-type gat
# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer transformer --gnn-type gat
# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer deepset --gnn-type gat

# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer setrep --gnn-type gcn
# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer transformer --gnn-type gcn
# python scripts/molnet_test_runner.py esol srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer deepset --gnn-type gcn

# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer setrep
# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer transformer
# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer deepset

# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer setrep --gnn-type gat
# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer transformer --gnn-type gat
# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer deepset --gnn-type gat

# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer setrep --gnn-type gcn
# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer transformer --gnn-type gcn
# python scripts/molnet_test_runner.py lipo srgnn --monitor loss --splitter custom-scaffold --n 4 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --set-layer deepset --gnn-type gcn

# python scripts/molnet_test_runner.py esol gnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --gnn-type gat
# python scripts/molnet_test_runner.py lipo gnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --gnn-type gat
# python scripts/molnet_test_runner.py bbbp gnn --monitor loss --task-type classification --splitter custom-scaffold --n 3 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --gnn-type gat

python scripts/molnet_test_runner.py esol gnn --monitor loss --splitter custom-scaffold --n 3 --max-epochs 150 --project srgnn_comparisons --n-hidden-channels 64 --n-hidden-channels 64 --n-layers 8 --gnn-type gat --set-layer transformer