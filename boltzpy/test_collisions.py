import pytest
import os
import h5py
import numpy as np

import boltzpy as bp
import boltzpy.helpers.tests as test_helper
from boltzpy.test_model import MODELS

FILE = test_helper.DIRECTORY + 'Collisions.hdf5'
RULES = dict()
COLLISIONS = dict()

# TODO remove this, when moving Collisions into sv/model:
COLLISIONS["2D_small/Collisions"] = bp.Collisions(
    MODELS["2D_small/Model"].collision_relations,
    MODELS["2D_small/Model"].collision_weights)
COLLISIONS["equalMass/Collisions"] = bp.Collisions(
    MODELS["equalMass/Model"].collision_relations,
    MODELS["equalMass/Model"].collision_weights)


def setup_file(file_address=FILE):
    if file_address == FILE:
        reply = input("You are about to reset the collisions test file. "
                      "Are you Sure? (yes, no)\n")
        if reply != "yes":
            print("ABORTED")
            return

    file = h5py.File(file_address, mode="w")
    for (key, item) in COLLISIONS.items():
        assert isinstance(item, bp.Collisions)
        file.create_group(key)
        item.save(file[key])
    return


#############################
#           Tests           #
#############################
def test_file_exists():
    assert os.path.exists(FILE), (
        "The test file {} is missing.".format(FILE))


def test_setup_creates_same_file():
    setup_file(test_helper.TMP_FILE)
    test_helper.assert_files_are_equal([FILE, test_helper.TMP_FILE])
    os.remove(test_helper.TMP_FILE)
    return


@pytest.mark.parametrize("key", COLLISIONS.keys())
def test_hdf5_groups_exist(key):
    file = h5py.File(FILE, mode="r")
    assert key in file.keys(), (
        "The group {} is missing in the test file-".format(key))


@pytest.mark.parametrize("key", COLLISIONS.keys())
def test_load_from_file(key):
    file = h5py.File(FILE, mode="r")
    hdf_group = file[key]
    old = bp.Collisions.load(hdf_group)
    new = COLLISIONS[key]
    assert isinstance(old, bp.Collisions)
    assert isinstance(new, bp.Collisions)
    assert old == new, (
        "\n{}\nis not equal to\n\n{}".format(old, new)
    )

#
# @pytest.mark.parametrize("tc", bp_t.CASES)
# def test_collisions(tc):
#     # Compute Output in temporary file
#     sim = bp.Simulation.load(file=tc.file)
#     old_coll = sim.coll
#     # new collisions are generated in the testcases already
#     new_coll = tc.coll
#     # sort collisions, to ignore different orders
#     old_coll.sort()
#     new_coll.sort()
#     # compare results
#     assert old_coll.size == new_coll.size
#     for (c, coll) in enumerate(old_coll.relations):
#         assert sorted(coll) == sorted(new_coll.relations[c])
#         colliding_species = {tc.sv.get_specimen(idx) for idx in coll}
#         if len(colliding_species) == 1:
#             permutations = bp.Collisions.INTRASPECIES_PERMUTATION
#         else:
#             permutations = bp.Collisions.INTERSPECIES_PERMUTATION
#         assert any(np.all(coll[p] == new_coll.relations[c])
#                    for p in permutations)
#         assert old_coll.weights[c] == new_coll.weights[c]
#     return

# Todo Test that complete >= convergent
# def test_issubset():
#     tc1 = bp_t.CASES[2]
#     tc2 = bp_t.CASES[3]
#     assert tc2.coll.issubset(tc1.coll)
#     assert not tc1.coll.issubset(tc2.coll)
#     return
