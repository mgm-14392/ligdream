from torch.utils.data import Dataset
import generators
import h5py


class CustomDataset(Dataset):
    # A pytorch dataset class for holding data for a text classification task.
    def __init__(self, filename_smiles, filename_hdf5):
        # Opening the file and storing its contents in a list
        with open(filename_smiles) as f:
            lines = f.readlines()

        self.lines = lines
        self.filename_hdf5 = filename_hdf5
        self._h5_gen = None

    # Process functions
    def line_mapper(self, line, filename_hdf5):
        # We only have the text in the file for this case
        mol = generators.coords2grid(line, filename_hdf5)
        caption, length = generators.smiles_to_np(line)
        return mol, caption, length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        if self._h5_gen is None:
            self._h5_gen = h5py.File(self.filename_hdf5, 'r')
        return self.line_mapper(self.lines[index], self._h5_gen)
