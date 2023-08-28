############
# ADME
############
<<<<<<< HEAD
python scripts/molnet_test_runner.py ocelot srgnn  --monitor loss --splitter scaffold --n 5 --max-epochs 150 --project ocelot-final --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task homo
python scripts/molnet_test_runner.py ocelot gnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --project ocelot-final --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --task homo
python scripts/molnet_test_runner.py ocelot msr1 --monitor loss --splitter scaffold --n 5 --max-epochs 250 --project ocelot-final --n-hidden-sets 64 --n-elements 4 --task homo
python scripts/molnet_test_runner.py ocelot msr2 --monitor loss --splitter scaffold --n 5 --max-epochs 250 --project ocelot-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task homo
=======
# python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128 --project ocelot-final --task vie
# python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128 --project ocelot-final --task homo
# python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128 --project ocelot-final --task ar1
python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128 --project ocelot-final --task hl
python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128 --project ocelot-final --task lumo


# python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 1 --max-epochs 150 --batch-size 128 --project ocelot-final --task vie
# python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 1 --max-epochs 150 --n-hidden-sets 128 --n-elements 64 --n-hidden-channels 128 --n-hidden-channels 64 --n-layers 8 --project ocelot-final --task vie
# python scripts/molnet_test_runner.py ocelot gnn --monitor loss --splitter scaffold --n 1 --max-epochs 150 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128 --project ocelot-final --task vie

# python scripts/molnet_test_runner.py ocelot srgnn --monitor loss --splitter scaffold --n 1 --max-epochs 300 --n-hidden-channels 64 --n-hidden-channels 32 --n-layers 4 --batch-size 128 --project ocelot-final --task vie
# python scripts/molnet_test_runner.py ocelot srgnn  --monitor loss --splitter scaffold --n 1 --max-epochs 150 --n-layers 4 --batch-size 128 --project ocelot-final --variant defaults --task vie
# python scripts/molnet_test_runner.py ocelot srgnn  --monitor loss --splitter scaffold --n 1 --max-epochs 300 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128 --project ocelot-final --task vie
# python scripts/molnet_test_runner.py ocelot srgnn  --monitor loss --splitter scaffold --n 1 --max-epochs 300 --n-hidden-channels 64 --n-hidden-channels 32 --batch-size 128 --project ocelot-final --learning-rate 0.0005 --task vie



# python scripts/molnet_test_runner.py ocelot gnn --monitor loss --splitter scaffold --n 5 --max-epochs 150 --project ocelot-final --n-hidden-channels 64 --n-hidden-channels 32 --n-layers 4 --task homo
# python scripts/molnet_test_runner.py ocelot msr1 --monitor loss --splitter scaffold --n 5 --max-epochs 250 --project ocelot-final --n-hidden-sets 64 --n-elements 4 --task homo
# python scripts/molnet_test_runner.py ocelot msr2 --monitor loss --splitter scaffold --n 5 --max-epochs 250 --project ocelot-final --n-hidden-sets 64 --n-hidden-sets 64 --n-elements 4 --n-elements 4 --task homo
>>>>>>> b08c3c88662fcc03bcc47b26c5c1e43693d91b32
