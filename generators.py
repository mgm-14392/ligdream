# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
# Captioning network for BycycleGAN (https://github.com/junyanz/BicycleGAN) and
# Modified from Ligdream (https://github.com/compsciencelab/ligdream)
# Modification of the original code to use libmolgrid for input preparation 8/04/22

import molgrid
from molgrid import Coords2GridFunction
import baseoptions
import numpy as np
import torch
import pybel
from rdkit import Chem


args = baseoptions.BaseOptions().create_parser()


def makecanonical(text):
    """Make smiles canonical"""

    mol = Chem.MolFromSmiles(text)
    if not mol:
        raise ValueError("Failed to parse molecule '{}'".format(mol))
    sstring = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)
    return sstring


def smiles_to_np(text):
    """ convert SMILES into tensor """

    vocab_list_i = ["pad", "start", "end",
                    "C", "c", "N", "n", "S", "s", "P", "O", "o",
                    "B", "F", "I",
                    "X", "Y", "Z",
                    "1", "2", "3", "4", "5", "6",
                    "#", "=", "-", "(", ")"
                    ]
    vocab_i2c_v1_i = {i: x for i, x in enumerate(vocab_list_i)}
    vocab_c2i_v1_i = {vocab_i2c_v1_i[i]: i for i in vocab_i2c_v1_i}

    strings = np.zeros(72, dtype='uint8')

    sstring = makecanonical(text)

    sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").\
        replace("Br", "Z").replace("[PH]", "P").\
        replace("[SH]", "S").replace("[S]", "S")

    try:
        vals = [1] + [vocab_c2i_v1_i[xchar] for xchar in sstring] + [2]
    except KeyError:
        raise ValueError(("Unkown SMILES tokens: {} in string '{}'."
                          .format(", ".join([x for x in sstring if x not in vocab_c2i_v1_i]),
                                  sstring)))

    end_token = vals.index(2)

    strings[0: len(vals)] = vals

    stensor = torch.from_numpy(strings)
    stensor = stensor.type(torch.LongTensor)
    return stensor, end_token + 1


def get_mol(smile, h5file):
    dataset = h5file[smile]
    endstring = ''
    for line in dataset:
        endstring += line.decode()
    return endstring


def coords2grid(text, hdf5_file, resolution=args['grid_resol'], dimension=args['grid_dim']):

    smiles = text.strip()
    g_mol = get_mol(smiles, hdf5_file)
    mol = pybel.readstring('sdf', g_mol)

    crds = molgrid.CoordinateSet(mol.OBMol, molgrid.defaultGninaLigandTyper)
    crds.make_vector_types()
    center = tuple(crds.center())
    coordinates = torch.tensor(crds.coords.tonumpy())
    radii = torch.tensor(crds.radii.tonumpy())
    types = torch.tensor(crds.type_vector.tonumpy())

    # initialize gridMaker
    gmaker = molgrid.GridMaker(resolution=resolution, dimension=dimension, binary=False,
                               radius_type_indexed=False, radius_scale=1.0, gaussian_radius_multiple=1.0)

    grid = Coords2GridFunction.apply(gmaker, center, coordinates, types, radii)
    return grid
