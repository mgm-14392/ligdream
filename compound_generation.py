# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
# Captioning network for BycycleGAN (https://github.com/junyanz/BicycleGAN) and
# Modified from Ligdream (https://github.com/compsciencelab/ligdream)
# Modification of the original code to use libmolgrid for input preparation 8/04/22

from networks import EncoderCNN_v3, DecoderRNN, VAE
from generators import makecanonical, get_mol, coords2grid
from decoding import decode_smiles
from torch.autograd import Variable
import molgrid
from molgrid import Coords2GridFunction
import torch
import pybel
from rdkit import Chem
import sys
import h5py


def filter_unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Check for duplicates and filter out invalids


def get_smi_3D_voxels(smiles, filename_hdf5=None, evaluate=False):
    # SMILES to 3D with openbabel
    if evaluate:

        smiles = makecanonical(smiles)
        mol = pybel.readstring('smi', smiles)
        mol.addh()

        ff = pybel._forcefields["mmff94"]
        success = ff.Setup(mol.OBMol)
        if not success:
            ff = pybel._forcefields["uff"]
            success = ff.Setup(mol.OBMol)
            if not success:
                sys.exit("Cannot set up forcefield")

        ff.ConjugateGradients(100, 1.0e-3)  # optimize geometry
        ff.WeightedRotorSearch(100, 25)  # generate conformers
        ff.ConjugateGradients(250, 1.0e-4)
        ff.GetCoordinates(mol.OBMol)

        # get 3D coords
        #crds = molgrid.CoordinateSet(mol.OBMol, molgrid.defaultGninaLigandTyper)
        crds = molgrid.CoordinateSet(mol.OBMol)

    else:
        hdf5_file = h5py.File(filename_hdf5, 'r')
        g_mol = get_mol(smiles, hdf5_file)
        mol = pybel.readstring('sdf', g_mol)
        crds = molgrid.CoordinateSet(mol)


    gmaker = molgrid.GridMaker(resolution=1, dimension=23, binary=False,
                               radius_type_indexed=False, radius_scale=1.0,
                               gaussian_radius_multiple=1.0)

    dims = gmaker.grid_dimensions(molgrid.defaultGninaLigandTyper.num_types())
    gridtensor = torch.zeros(dims, dtype=torch.float32)
    gmaker.forward(crds.center(), crds, gridtensor)

    #crds.make_vector_types()
    #center = tuple(crds.center())
    #coordinates = torch.tensor(crds.coords.tonumpy())
    #radii = torch.tensor(crds.radii.tonumpy())
    #types = torch.tensor(crds.type_vector.tonumpy())

    # initialize gridMaker
    #gmaker = molgrid.GridMaker(resolution=1, dimension=23, binary=False,
    #                           radius_type_indexed=False, radius_scale=1.0,
    #                           gaussian_radius_multiple=1.0)

    #grid = Coords2GridFunction.apply(gmaker, center, coordinates, types, radii)
    return gridtensor


class CompoundGenerator:
    def __init__(self, use_cuda=True):

        self.use_cuda = False
        self.encoder = EncoderCNN_v3(15)
        self.decoder = DecoderRNN(512, 1024, 29, 1)
        self.vae_model = VAE()

        self.vae_model.eval()
        self.encoder.eval()
        self.decoder.eval()

        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
            self.vae_model.cuda()
            self.use_cuda = True

    def load_weight(self, vae_weights, encoder_weights, decoder_weights):
        """
        Load the weights of the models.
        :param vae_weights: str - VAE model weights path
        :param encoder_weights: str - captioning model encoder weights path
        :param decoder_weights: str - captioning model decoder model weights path
        :return: None
        """
        self.vae_model.load_state_dict(vae_weights['model_state_dict'])
        self.vae_model.eval()
        self.encoder.load_state_dict(encoder_weights['model_state_dict'])
        self.encoder.eval()
        self.decoder.load_state_dict(decoder_weights['model_state_dict'])
        self.decoder.eval()

    def caption_shape(self, in_shapes, probab=False):
        """
        Generates SMILES representation from in_shapes
        """
        embedding = self.encoder(in_shapes)
        if probab:
            captions = self.decoder.sample_prob(embedding)
        else:
            captions = self.decoder.sample(embedding)

        captions = torch.stack(captions, 1)
        if self.use_cuda:
            captions = captions.cpu().data.numpy()
        else:
            captions = captions.data.numpy()
        return decode_smiles(captions)

    def generate_molecules(self, shape, n_attemps=1, probab=False, filter_unique_valid=True):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param probab: boolean - use probabilistic decoding
        :param filter_unique_valid: boolean - filter for valid and unique molecules
        :return: list of RDKit molecules.
        """

        shape_input = shape
        if self.use_cuda:
            shape_input = shape_input.cuda()

        shape_input = shape_input.unsqueeze(0).repeat(n_attemps, 1, 1, 1, 1)

        shape_input = Variable(shape_input)

        recoded_shapes, _, _ = self.vae_model(shape_input)
        smiles = self.caption_shape(recoded_shapes, probab=probab)
        if filter_unique_valid:
            return filter_unique_canonical(smiles)
        return [Chem.MolFromSmiles(x) for x in smiles]

if __name__ == '__main__':
    filename_hdf5 = ''
    smiles = ''
    vae_weights = ''
    encoder_weights = ''
    decoder_weights = ''

    CompoundGenerator.load_weight(vae_weights, encoder_weights, decoder_weights)
    shape = get_smi_3D_voxels(smiles, filename_hdf5, False)
    generated_smiles = CompoundGenerator.generate_molecules(shape)



