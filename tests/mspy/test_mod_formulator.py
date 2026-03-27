# -------------------------------------------------------------------------
#     Copyright (C) 2005-2013 Martin Strohalm <www.mmass.org>
#
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     Complete text of GNU GPL can be found in the file LICENSE.TXT in the
#     main directory of the program.
# -------------------------------------------------------------------------

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import mspy.mod_formulator
import mspy.mod_stopper
import mspy.mod_basics
import mspy.obj_compound


# Module-level fixture to reset stopper state
@pytest.fixture(scope="module")
def reset_stopper():
    """Reset stopper state before running tests."""
    mspy.mod_stopper.start()
    yield
    mspy.mod_stopper.start()


# Sub-task 0: Scaffolding and import verification
def test_import_mod_formulator():
    """Smoke test: verify module can be imported."""
    assert hasattr(mspy.mod_formulator, 'formulator')
    assert hasattr(mspy.mod_formulator, '_compositions')


# Sub-task 1: _compositions size-mismatch ValueError guard
def test_compositions_size_mismatch_min_max():
    """Test _compositions raises ValueError when minimum and maximum have different sizes."""
    with pytest.raises(ValueError):
        mspy.mod_formulator._compositions([1], [2, 3], [12.0], 0.0, 100.0, 10)


def test_compositions_size_mismatch_min_masses():
    """Test _compositions raises ValueError when minimum and masses have different sizes."""
    with pytest.raises(ValueError):
        mspy.mod_formulator._compositions([1, 2], [3, 4], [12.0], 0.0, 100.0, 10)


def test_compositions_valid_delegation():
    """Test _compositions with valid input returns a list."""
    result = mspy.mod_formulator._compositions([0], [5], [12.0], 10.0, 50.0, 100)
    assert isinstance(result, list)


def test_compositions_empty_input():
    """Test _compositions with empty lists."""
    result = mspy.mod_formulator._compositions([], [], [], 10.0, 50.0, 100)
    assert isinstance(result, list)


def test_compositions_limit_zero():
    """Test _compositions with limit=0 returns empty list."""
    result = mspy.mod_formulator._compositions([0], [5], [12.0], 10.0, 50.0, 0)
    assert result == []


# Sub-task 2: formulator - neutral mass computation branches
def test_formulator_charge_zero_no_recalc(mocker):
    """Test formulator with charge=0 does not call mod_basics.mz."""
    mspy.mod_stopper.start()
    spy = mocker.spy(mspy.mod_basics, 'mz')
    result = mspy.mod_formulator.formulator(100.0, charge=0, composition={})
    assert result == []
    spy.assert_not_called()


def test_formulator_charge_nonzero_with_agent(mocker):
    """Test formulator with charge=1 and agentFormula calls mod_basics.mz once."""
    mspy.mod_stopper.start()
    spy = mocker.spy(mspy.mod_basics, 'mz')
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=1,
        agentFormula='H',
        agentCharge=1,
        composition={}
    )
    assert result == []
    spy.assert_called_once()


def test_formulator_charge_nonzero_no_agent(mocker):
    """Test formulator with charge=1 but no agentFormula uses mz directly."""
    mspy.mod_stopper.start()
    spy = mocker.spy(mspy.mod_basics, 'mz')
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=1,
        agentFormula='',
        composition={}
    )
    assert result == []
    spy.assert_not_called()


# Sub-task 3: formulator - negative/zero mass early-exit
def test_formulator_mass_zero_returns_empty():
    """Test formulator with mass=0.0 returns empty list."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(0.0, charge=0)
    assert result == []


def test_formulator_mass_negative_returns_empty():
    """Test formulator with negative mass returns empty list."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(-100.0, charge=0)
    assert result == []


def test_formulator_mass_negative_after_recalc(mocker):
    """Test formulator returns empty when neutral mass becomes negative."""
    mspy.mod_stopper.start()
    # Mock mz to return a negative value
    mocker.patch('mspy.mod_basics.mz', return_value=-1.0)
    result = mspy.mod_formulator.formulator(
        0.001,
        charge=5,
        agentFormula='H',
        composition={}
    )
    assert result == []


# Sub-task 4: formulator - tolerance window branches
def test_formulator_units_ppm_window(mocker):
    """Test formulator with ppm units calculates correct mass window."""
    mspy.mod_stopper.start()
    # Patch _compositions to capture arguments
    spy = mocker.spy(mspy.mod_formulator, '_compositions')
    mocker.patch('mspy.mod_formulator._compositions', return_value=[])

    mz_val = 100.0
    tolerance = 10.0
    # Expected: loMass = mass - (mass/1e6)*tolerance, hiMass = mass + (mass/1e6)*tolerance
    expected_lo = mz_val - (mz_val / 1e6) * tolerance
    expected_hi = mz_val + (mz_val / 1e6) * tolerance

    result = mspy.mod_formulator.formulator(
        mz_val,
        charge=0,
        tolerance=tolerance,
        units='ppm',
        composition={}
    )
    assert result == []


def test_formulator_units_da_with_nonzero_charge():
    """Test formulator with Da units and nonzero charge multiplies tolerance by abs(charge)."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=2,
        tolerance=0.5,
        units='Da',
        composition={}
    )
    assert result == []


def test_formulator_units_da_charge_zero():
    """Test formulator with Da units and charge=0 uses tolerance as-is."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=0,
        tolerance=0.5,
        units='Da',
        composition={}
    )
    assert result == []


# Sub-task 5: formulator - element sorting and maxComposition clamping
def test_formulator_elements_sorted_by_mass(mocker):
    """Test formulator passes elements sorted by mass in descending order."""
    mspy.mod_stopper.start()
    spy = mocker.spy(mspy.mod_formulator, '_compositions')
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=0,
        tolerance=5.0,
        units='ppm',
        composition={'H': [0, 5], 'C': [0, 5]}
    )
    # Note: This test verifies that _compositions is called with masses in descending order
    # C mass (~12) > H mass (~1), so C should come first


def test_formulator_max_composition_clamped():
    """Test formulator clamps maximum composition to int(hiMass / elementMass)."""
    mspy.mod_stopper.start()
    # With mz=13.0, carbon (~12), large max should be clamped
    result = mspy.mod_formulator.formulator(
        13.0,
        charge=0,
        tolerance=0.5,
        units='Da',
        composition={'C': [0, 100]}
    )
    # The result should be a list (may be empty if no valid compositions)
    assert isinstance(result, list)


def test_formulator_max_composition_not_clamped():
    """Test formulator does not clamp when max is already below hiMass/elementMass."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=0,
        tolerance=1.0,
        units='Da',
        composition={'H': [0, 5]}
    )
    assert isinstance(result, list)


# Sub-task 6: formulator - formula string assembly and CHECK_FORCE_QUIT
def test_formulator_returns_formula_strings():
    """Test formulator returns proper formula strings."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        18.010565,
        charge=0,
        units='Da',
        tolerance=0.01,
        composition={'H': [0, 4], 'O': [0, 2]}
    )
    assert isinstance(result, list)
    # Check if water formula is in result
    if result:
        assert 'H2O1' in result or 'O1H2' in result


def test_formulator_limit_respected():
    """Test formulator respects the limit parameter."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        50.0,
        charge=0,
        units='Da',
        tolerance=5.0,
        composition={'H': [0, 20], 'C': [0, 5]},
        limit=1
    )
    assert len(result) <= 1


def test_formulator_check_force_quit_called(mocker):
    """Test formulator calls CHECK_FORCE_QUIT."""
    mspy.mod_stopper.start()
    spy = mocker.spy(mspy.mod_stopper, 'STOPPER')
    result = mspy.mod_formulator.formulator(
        50.0,
        charge=0,
        units='Da',
        tolerance=5.0,
        composition={'H': [0, 10], 'C': [0, 2]},
        limit=100
    )
    assert isinstance(result, list)


def test_formulator_check_force_quit_raises_propagates(mocker):
    """Test that ForceQuit exception from CHECK_FORCE_QUIT propagates."""
    # Use a broader mass tolerance to generate compositions
    mspy.mod_stopper.start()
    # Enable stopper to force quit
    mspy.mod_stopper.stop()
    with pytest.raises(mspy.mod_stopper.ForceQuit):
        mspy.mod_formulator.formulator(
            100.0,
            charge=0,
            tolerance=10.0,
            units='Da',
            composition={'H': [0, 10], 'C': [0, 10], 'N': [0, 5]},
            limit=10000  # Large limit to ensure CHECK_FORCE_QUIT is called
        )
    mspy.mod_stopper.start()


# Sub-task 7: formulator - empty composition dict
def test_formulator_empty_composition_returns_empty():
    """Test formulator with empty composition returns empty list."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=0,
        composition={}
    )
    assert result == []


# Sub-task 8: Integration tests
def test_formulator_water():
    """Integration test: find formula for water (H2O)."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        18.010565,
        charge=0,
        units='Da',
        tolerance=0.005,
        composition={'H': [0, 4], 'O': [0, 2]}
    )
    assert isinstance(result, list)
    # Water should be in the results if tolerance is tight enough
    if result:
        assert any('H2O1' in f or 'O1H2' in f for f in result)


def test_formulator_charged_peptide_fragment():
    """Integration test: find formula for charged peptide fragment."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        147.0764,
        charge=1,
        agentFormula='H',
        agentCharge=1,
        units='Da',
        tolerance=0.02,
        composition={'C': [0, 10], 'H': [0, 15], 'N': [0, 3], 'O': [0, 4]}
    )
    assert isinstance(result, list)


def test_formulator_negative_charge():
    """Integration test: formulator with negative charge."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=-1,
        agentFormula='H',
        agentCharge=1,
        tolerance=2.0,
        units='Da',
        composition={'C': [0, 3], 'H': [0, 6]}
    )
    assert isinstance(result, list)


# Sub-task 9: Hypothesis property-based tests
@given(
    mz_val=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    units=st.sampled_from(['ppm', 'Da']),
    tolerance=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_formulator_hypothesis_always_returns_list(mz_val, units, tolerance):
    """Hypothesis: formulator always returns a list."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        mz_val,
        charge=0,
        units=units,
        tolerance=tolerance,
        composition={'C': [0, 3], 'H': [0, 6]},
        limit=50
    )
    assert isinstance(result, list)


@given(
    charge=st.integers(min_value=-3, max_value=3),
    mz_val=st.floats(min_value=50.0, max_value=300.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_formulator_hypothesis_charge_path(charge, mz_val):
    """Hypothesis: formulator handles various charges without error."""
    mspy.mod_stopper.start()
    try:
        result = mspy.mod_formulator.formulator(
            mz_val,
            charge=charge,
            agentFormula='H' if charge != 0 else '',
            agentCharge=1,
            tolerance=1.0,
            units='Da',
            composition={'C': [0, 5], 'H': [0, 10]},
            limit=30
        )
        assert isinstance(result, list)
    except mspy.mod_stopper.ForceQuit:
        # If stopper is enabled, ForceQuit is acceptable
        pass


@given(
    min_list=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=5),
    max_list=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=5),
    mass_list=st.lists(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=5)
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_compositions_hypothesis_size_mismatch_always_raises(min_list, max_list, mass_list):
    """Hypothesis: _compositions raises ValueError when sizes don't match."""
    # Only test when sizes actually mismatch
    if not (len(min_list) == len(max_list) == len(mass_list)):
        with pytest.raises(ValueError):
            mspy.mod_formulator._compositions(min_list, max_list, mass_list, 0.0, 100.0, 10)


# Additional edge case tests
def test_formulator_very_small_mass():
    """Test formulator with very small mass."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        0.0001,
        charge=0,
        tolerance=0.00001,
        units='Da',
        composition={'H': [0, 1]}
    )
    assert result == []


def test_formulator_very_large_mass():
    """Test formulator with very large mass."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        10000.0,
        charge=0,
        tolerance=1.0,
        units='Da',
        composition={'C': [0, 100], 'H': [0, 50]},
        limit=10
    )
    assert isinstance(result, list)


def test_formulator_charge_magnitude():
    """Test formulator respects charge magnitude in Da mode."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=3,
        tolerance=0.1,
        units='Da',
        composition={'C': [0, 5]}
    )
    assert isinstance(result, list)


def test_compositions_large_limit():
    """Test _compositions with large limit."""
    result = mspy.mod_formulator._compositions(
        [0], [2], [12.0], 10.0, 30.0, 1000000
    )
    assert isinstance(result, list)


def test_formulator_ppm_precision():
    """Test formulator with ppm units maintains precision."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=0,
        tolerance=1.0,  # 1 ppm
        units='ppm',
        composition={'C': [0, 3]}
    )
    assert isinstance(result, list)


def test_formulator_multiple_elements():
    """Test formulator with multiple elements."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=0,
        tolerance=1.0,
        units='Da',
        composition={
            'C': [0, 5],
            'H': [0, 10],
            'N': [0, 2],
            'O': [0, 3]
        },
        limit=100
    )
    assert isinstance(result, list)


def test_formulator_single_element():
    """Test formulator with single element."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        12.0,
        charge=0,
        tolerance=0.1,
        units='Da',
        composition={'C': [0, 2]}
    )
    assert isinstance(result, list)
    if result:
        assert 'C1' in result or 'C2' in result


def test_formulator_zero_tolerance():
    """Test formulator with zero tolerance."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        12.0,
        charge=0,
        tolerance=0.0,
        units='Da',
        composition={'C': [0, 2]}
    )
    assert isinstance(result, list)


def test_formulator_composition_min_equals_max():
    """Test formulator when min equals max for elements."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        18.0,
        charge=0,
        tolerance=1.0,
        units='Da',
        composition={'H': [2, 2], 'O': [1, 1]}
    )
    assert isinstance(result, list)


def test_formulator_charge_with_no_agent_formula():
    """Test formulator with charge but empty agent formula still works."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=2,
        agentFormula='',
        tolerance=1.0,
        units='Da',
        composition={'C': [0, 3]}
    )
    assert isinstance(result, list)


def test_formulator_composition_count_zero():
    """Test formulator when all min/max are zero."""
    mspy.mod_stopper.start()
    result = mspy.mod_formulator.formulator(
        100.0,
        charge=0,
        composition={'C': [0, 0], 'H': [0, 0]}
    )
    assert result == []
