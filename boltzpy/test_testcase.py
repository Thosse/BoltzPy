
import boltzpy as bp
import boltzpy.testcase as bp_t
import pytest
import numpy as np
import os
import h5py


@pytest.mark.parametrize("tc", bp_t.CASES)
def test_file_exists(tc):
    assert isinstance(tc, bp_t.TestCase)
    assert os.path.exists(tc.file_address)


@pytest.mark.parametrize("tc", bp_t.CASES)
def test_file_initializes_the_correct_simulation(tc):
    sim = bp.Simulation.load(tc.file_address)
    assert tc.s == sim.s
    assert tc.t == sim.t
    assert tc.p == sim.p
    assert tc.sv == sim.sv
    assert tc.geometry == sim.geometry
    assert tc.scheme == sim.scheme
    assert np.all(tc.output_parameters == sim.output_parameters)
    assert sim.__eq__(tc, True)
    assert not tc.__eq__(sim, False)



@pytest.mark.parametrize("tc", bp_t.CASES)
def test_new_file_is_equal_to_current_one(tc):
    assert isinstance(tc, bp_t.TestCase)
    # TODO move this into prerun of pytest
    # remove new_Address, if it exists already
    if os.path.exists(tc.temporary_file):
        os.remove(tc.temporary_file)

    # create new file
    new_file = tc.create_file(tc.temporary_file)
    old_file = h5py.File(tc.file_address, mode='r')
    # Todo Check ALL keys, not just results
    for species_name in new_file["results"].keys():
        for output in new_file["results"][species_name].keys():
            key = "results/{}/{}".format(species_name, output)
            old_results = old_file[key][()]
            new_results = new_file[key][()]
            assert old_results.shape == new_results.shape
            assert np.array_equal(old_results, new_results)
            # except AssertionError:
            #     print("Update failed: ", tc.file_address)
            #     print("\tDifferences found in:",
            #           "\nspecies: ", species_name,
            #           "\noutput: ", output)

    os.remove(tc.temporary_file)
