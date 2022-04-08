import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data directory
        self.parser.add_argument("-i", "--input", required=True, help="Path to input file with SMILES")
        self.parser.add_argument("-o", "--output_dir", required=True, help="Name of directory to save models")

        # ligand grid parameters
        self.parser.add_argument("--grid_resol", type=int, default=1.0, help="Grid resolution")
        self.parser.add_argument("--grid_dim", type=int, default=23.0, help="Grid dimension")
        self.parser.add_argument('--rotation', action='store_true', help="To not rotate data don't include --rotation "
                                                                         "in args")

        # training
        self.parser.add_argument("--epochs", type=int, default=7, help="number of epochs")
        self.parser.add_argument("--batch_size", type=int, default=100, help="input batch size")
        self.parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
        self.parser.add_argument("--start_cap", type=int, default=2, help="iteration to start training caption")

        # hyperparameters
        self.parser.add_argument("--lr_cap", type=float, default=0.001, help="initial learning rate for adam")
        self.parser.add_argument("--lr_VAE", type=float, default=0.0009, help="initial learning rate for adam")

        # GPU ids
        self.parser.add_argument("--GPUIDS", required=True, nargs="+", type=int,
                                 help="GPU ids available, example 0 1 2")

    def create_parser(self):
        return vars(self.parser.parse_args())


args = BaseOptions().create_parser()

print(args)
