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
from hypothesis import given, strategies as st
import obj_peak
import mod_basics


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def basic_peak():
    """Create a basic peak with minimal arguments."""
    return obj_peak.peak(mz=500.0, ai=100.0, base=10.0, charge=1)


@pytest.fixture
def peak_with_fwhm():
    """Create a peak with FWHM for resolution calculation."""
    return obj_peak.peak(mz=500.0, ai=100.0, base=10.0, charge=1, fwhm=0.5)


@pytest.fixture
def peak_no_charge():
    """Create a peak without charge."""
    return obj_peak.peak(mz=500.0, ai=100.0, base=10.0)


@pytest.fixture
def peak_with_attributes():
    """Create a peak with extra attributes."""
    return obj_peak.peak(mz=500.0, ai=100.0, base=10.0, charge=1, custom_attr='value', another=42)


# ============================================================================
# __init__ TESTS - Coverage of B1 and B2 branches
# ============================================================================

class TestPeakInit:
    """Test peak initialization."""

    def test_init_basic_attributes(self):
        """Test basic initialization of peak attributes."""
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0, sn=50.0, charge=2, isotope='M+1', fwhm=0.5, group='test')
        assert p.mz == 500.0
        assert p.ai == 100.0
        assert p.base == 10.0
        assert p.sn == 50.0
        assert p.charge == 2
        assert p.isotope == 'M+1'
        assert p.fwhm == 0.5
        assert p.group == 'test'

    def test_init_converts_mz_to_float(self):
        """Test that mz is converted to float."""
        p = obj_peak.peak(mz=500)
        assert isinstance(p.mz, float)
        assert p.mz == 500.0

    def test_init_converts_ai_to_float(self):
        """Test that ai is converted to float."""
        p = obj_peak.peak(mz=500.0, ai=100)
        assert isinstance(p.ai, float)
        assert p.ai == 100.0

    def test_init_converts_base_to_float(self):
        """Test that base is converted to float."""
        p = obj_peak.peak(mz=500.0, base=10)
        assert isinstance(p.base, float)
        assert p.base == 10.0

    def test_init_intensity_calculation(self):
        """Test intensity is calculated as ai - base."""
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0)
        assert p.intensity == 90.0

    def test_init_relative_intensity_ri_default(self):
        """Test relative intensity defaults to 1.0."""
        p = obj_peak.peak(mz=500.0)
        assert p.ri == 1.0

    def test_init_child_scan_number_default(self):
        """Test childScanNumber defaults to None."""
        p = obj_peak.peak(mz=500.0)
        assert p.childScanNumber is None

    def test_init_mass_buffer_default(self):
        """Test _mass buffer defaults to None."""
        p = obj_peak.peak(mz=500.0)
        assert p._mass is None

    # Branch B1: fwhm truthy - resolution is computed
    def test_init_fwhm_truthy_resolution_computed(self):
        """Test B1-true: when fwhm is truthy, resolution is computed."""
        p = obj_peak.peak(mz=500.0, fwhm=0.5)
        assert p.resolution is not None
        assert p.resolution == 500.0 / 0.5
        assert p.resolution == 1000.0

    # Branch B1: fwhm falsy (None) - resolution is None
    def test_init_fwhm_none_resolution_none(self):
        """Test B1-false: when fwhm is None, resolution is None."""
        p = obj_peak.peak(mz=500.0, fwhm=None)
        assert p.resolution is None

    # Branch B1: fwhm falsy (0.0) - resolution is None (0.0 is falsy)
    def test_init_fwhm_zero_resolution_none(self):
        """Test B1-false: when fwhm is 0.0 (falsy), resolution is None."""
        p = obj_peak.peak(mz=500.0, fwhm=0.0)
        assert p.resolution is None

    # Branch B2: extra kwargs - stored in attributes
    def test_init_extra_kwargs_stored_in_attributes(self):
        """Test B2-true: extra kwargs are stored in attributes dict."""
        p = obj_peak.peak(mz=500.0, custom='value', number=42, flag=True)
        assert p.attributes == {'custom': 'value', 'number': 42, 'flag': True}

    # Branch B2: no extra kwargs - empty attributes
    def test_init_no_extra_kwargs_empty_attributes(self):
        """Test B2-false: no extra kwargs results in empty attributes dict."""
        p = obj_peak.peak(mz=500.0, ai=100.0)
        assert p.attributes == {}

    def test_init_default_parameters(self):
        """Test default parameter values."""
        p = obj_peak.peak(mz=500.0)
        assert p.ai == 0.0
        assert p.base == 0.0
        assert p.sn is None
        assert p.charge is None
        assert p.isotope is None
        assert p.fwhm is None
        assert p.group == ''


# ============================================================================
# reset() TESTS - Coverage of B3 branch
# ============================================================================

class TestPeakReset:
    """Test peak reset method."""

    # Branch B3: fwhm truthy - resolution recomputed
    def test_reset_fwhm_truthy_resolution_recomputed(self):
        """Test B3-true: when fwhm is truthy, reset recomputes resolution."""
        p = obj_peak.peak(mz=500.0, fwhm=0.5, ai=100.0, base=10.0)
        # Modify values
        p.mz = 600.0
        p.ai = 150.0
        p.base = 20.0
        p._mass = 999.0
        # Reset
        p.reset()
        # Resolution should be recomputed with new mz
        assert p.resolution == 600.0 / 0.5
        assert p.resolution == 1200.0

    # Branch B3: fwhm falsy (None) - resolution stays None
    def test_reset_fwhm_none_resolution_none(self):
        """Test B3-false: when fwhm is None, reset sets resolution to None."""
        p = obj_peak.peak(mz=500.0, fwhm=None, ai=100.0, base=10.0)
        p._mass = 999.0
        p.reset()
        assert p.resolution is None

    # Branch B3: fwhm falsy (0.0) - resolution stays None
    def test_reset_fwhm_zero_resolution_none(self):
        """Test B3-false: when fwhm is 0.0 (falsy), reset sets resolution to None."""
        p = obj_peak.peak(mz=500.0, fwhm=0.0, ai=100.0, base=10.0)
        p._mass = 999.0
        p.reset()
        assert p.resolution is None

    def test_reset_clears_mass_buffer(self):
        """Test that reset clears the _mass buffer."""
        p = obj_peak.peak(mz=500.0, charge=1)
        p._mass = 999.0
        p.reset()
        assert p._mass is None

    def test_reset_recalculates_intensity(self):
        """Test that reset recalculates intensity."""
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0)
        # Manually change ai and base
        p.ai = 150.0
        p.base = 30.0
        p.intensity = 999.0  # Wrong value
        # Reset
        p.reset()
        assert p.intensity == 150.0 - 30.0
        assert p.intensity == 120.0


# ============================================================================
# mass() TESTS - Coverage of B4 and B5 branches
# ============================================================================

class TestPeakMass:
    """Test peak mass getter method."""

    # Branch B4: charge == None - returns None
    def test_mass_charge_none_returns_none(self):
        """Test B4-true: when charge is None, mass() returns None."""
        p = obj_peak.peak(mz=500.0, charge=None)
        assert p.mass() is None

    # Branch B4: charge is set - continues to compute
    def test_mass_charge_positive_computes(self):
        """Test B4-false: when charge is positive, mass() computes mass."""
        p = obj_peak.peak(mz=500.0, charge=1)
        result = p.mass()
        assert result is not None
        assert isinstance(result, (float, tuple))

    def test_mass_charge_negative_computes(self):
        """Test B4-false: when charge is negative, mass() computes mass."""
        p = obj_peak.peak(mz=500.0, charge=-1)
        result = p.mass()
        assert result is not None

    def test_mass_charge_zero_computes(self):
        """Test B4-false: when charge is 0 (not None), mass() computes mass."""
        p = obj_peak.peak(mz=500.0, charge=0)
        result = p.mass()
        assert result is not None

    # Branch B5: _mass != None - returns cached value
    def test_mass_cached_value_returned(self):
        """Test B5-true: when _mass is not None, cached value is returned."""
        p = obj_peak.peak(mz=500.0, charge=1)
        p._mass = 999.0
        result = p.mass()
        assert result == 999.0

    # Branch B5: _mass is None - computes mass
    def test_mass_computed_when_buffer_empty(self):
        """Test B5-false: when _mass is None, mass is computed."""
        p = obj_peak.peak(mz=500.0, charge=1)
        assert p._mass is None
        result = p.mass()
        assert result is not None
        assert p._mass is not None
        assert p._mass == result

    def test_mass_integration_with_mod_basics_mz(self):
        """Test mass() integrates correctly with mod_basics.mz()."""
        p = obj_peak.peak(mz=500.0, charge=2)
        computed_mass = p.mass()

        # Verify it matches mod_basics.mz calculation
        expected = mod_basics.mz(500.0, 0, 2, agentFormula='H', agentCharge=1)
        assert computed_mass == expected

    def test_mass_caching_prevents_recomputation(self):
        """Test that mass() uses cache on second call."""
        p = obj_peak.peak(mz=500.0, charge=1)
        first_call = p.mass()
        p._mass = 12345.0  # Manually set cache
        second_call = p.mass()
        assert second_call == 12345.0
        assert first_call != second_call

    def test_mass_uses_correct_parameters(self):
        """Test mass() calls mod_basics.mz with correct parameters."""
        p = obj_peak.peak(mz=600.0, charge=3)
        result = p.mass()
        expected = mod_basics.mz(600.0, 0, 3, agentFormula='H', agentCharge=1)
        assert result == expected


# ============================================================================
# SETTER TESTS - Coverage of B6 and B7 branches
# ============================================================================

class TestPeakSetMz:
    """Test setmz method."""

    def test_setmz_updates_mz_value(self):
        """Test setmz updates the mz value."""
        p = obj_peak.peak(mz=500.0)
        p.setmz(600.0)
        assert p.mz == 600.0

    # Branch B6: fwhm truthy - resolution updated
    def test_setmz_fwhm_truthy_resolution_updated(self):
        """Test B6-true: when fwhm is truthy, setmz updates resolution."""
        p = obj_peak.peak(mz=500.0, fwhm=0.5)
        assert p.resolution == 500.0 / 0.5
        p.setmz(600.0)
        assert p.mz == 600.0
        assert p.resolution == 600.0 / 0.5
        assert p.resolution == 1200.0

    # Branch B6: fwhm falsy (None) - resolution stays None
    def test_setmz_fwhm_none_resolution_none(self):
        """Test B6-false: when fwhm is None, setmz doesn't set resolution."""
        p = obj_peak.peak(mz=500.0, fwhm=None)
        assert p.resolution is None
        p.setmz(600.0)
        assert p.mz == 600.0
        assert p.resolution is None

    # Branch B6: fwhm falsy (0.0) - resolution stays None
    def test_setmz_fwhm_zero_resolution_none(self):
        """Test B6-false: when fwhm is 0.0 (falsy), setmz doesn't set resolution."""
        p = obj_peak.peak(mz=500.0, fwhm=0.0)
        assert p.resolution is None
        p.setmz(600.0)
        assert p.mz == 600.0
        assert p.resolution is None

    def test_setmz_clears_mass_buffer(self):
        """Test setmz clears the _mass buffer."""
        p = obj_peak.peak(mz=500.0, charge=1)
        p._mass = 999.0
        p.setmz(600.0)
        assert p._mass is None

    def test_setmz_accepts_float_and_int(self):
        """Test setmz accepts both float and int values."""
        p = obj_peak.peak(mz=500)
        # setmz accepts the value as-is without converting
        p.setmz(600)
        assert p.mz == 600
        # When given float
        p.setmz(700.5)
        assert p.mz == 700.5


class TestPeakSetAi:
    """Test setai method."""

    def test_setai_updates_ai_value(self):
        """Test setai updates the ai value."""
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0)
        p.setai(150.0)
        assert p.ai == 150.0

    def test_setai_updates_intensity(self):
        """Test setai updates the intensity."""
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0)
        assert p.intensity == 90.0
        p.setai(150.0)
        assert p.intensity == 140.0


class TestPeakSetBase:
    """Test setbase method."""

    def test_setbase_updates_base_value(self):
        """Test setbase updates the base value."""
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0)
        p.setbase(20.0)
        assert p.base == 20.0

    def test_setbase_updates_intensity(self):
        """Test setbase updates the intensity."""
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0)
        assert p.intensity == 90.0
        p.setbase(20.0)
        assert p.intensity == 80.0


class TestPeakSetSn:
    """Test setsn method."""

    def test_setsn_updates_sn_value(self):
        """Test setsn updates the sn value."""
        p = obj_peak.peak(mz=500.0, sn=None)
        p.setsn(50.0)
        assert p.sn == 50.0

    def test_setsn_with_none(self):
        """Test setsn can set sn to None."""
        p = obj_peak.peak(mz=500.0, sn=50.0)
        p.setsn(None)
        assert p.sn is None


class TestPeakSetCharge:
    """Test setcharge method."""

    def test_setcharge_updates_charge_value(self):
        """Test setcharge updates the charge value."""
        p = obj_peak.peak(mz=500.0, charge=1)
        p.setcharge(2)
        assert p.charge == 2

    def test_setcharge_clears_mass_buffer(self):
        """Test setcharge clears the _mass buffer."""
        p = obj_peak.peak(mz=500.0, charge=1)
        p._mass = 999.0
        p.setcharge(2)
        assert p._mass is None

    def test_setcharge_to_none(self):
        """Test setcharge can set charge to None."""
        p = obj_peak.peak(mz=500.0, charge=1)
        p.setcharge(None)
        assert p.charge is None


class TestPeakSetIsotope:
    """Test setisotope method."""

    def test_setisotope_updates_isotope_value(self):
        """Test setisotope updates the isotope value."""
        p = obj_peak.peak(mz=500.0, isotope='M')
        p.setisotope('M+1')
        assert p.isotope == 'M+1'

    def test_setisotope_with_none(self):
        """Test setisotope can set isotope to None."""
        p = obj_peak.peak(mz=500.0, isotope='M+1')
        p.setisotope(None)
        assert p.isotope is None


class TestPeakSetFwhm:
    """Test setfwhm method."""

    def test_setfwhm_updates_fwhm_value(self):
        """Test setfwhm updates the fwhm value."""
        p = obj_peak.peak(mz=500.0, fwhm=0.5)
        p.setfwhm(1.0)
        assert p.fwhm == 1.0

    # Branch B7: new fwhm truthy - resolution updated
    def test_setfwhm_new_fwhm_truthy_resolution_updated(self):
        """Test B7-true: when new fwhm is truthy, setfwhm updates resolution."""
        p = obj_peak.peak(mz=500.0, fwhm=0.5)
        p.setfwhm(1.0)
        assert p.fwhm == 1.0
        assert p.resolution == 500.0 / 1.0
        assert p.resolution == 500.0

    # Branch B7: new fwhm falsy (None) - resolution set to None
    def test_setfwhm_new_fwhm_none_resolution_none(self):
        """Test B7-false: when new fwhm is None, setfwhm sets resolution to None."""
        p = obj_peak.peak(mz=500.0, fwhm=0.5)
        p.setfwhm(None)
        assert p.fwhm is None
        assert p.resolution is None

    # Branch B7: new fwhm falsy (0.0) - resolution set to None
    def test_setfwhm_new_fwhm_zero_resolution_none(self):
        """Test B7-false: when new fwhm is 0.0 (falsy), setfwhm sets resolution to None."""
        p = obj_peak.peak(mz=500.0, fwhm=0.5)
        p.setfwhm(0.0)
        assert p.fwhm == 0.0
        assert p.resolution is None


class TestPeakSetGroup:
    """Test setgroup method."""

    def test_setgroup_updates_group_value(self):
        """Test setgroup updates the group value."""
        p = obj_peak.peak(mz=500.0, group='')
        p.setgroup('isotope_group_1')
        assert p.group == 'isotope_group_1'

    def test_setgroup_with_empty_string(self):
        """Test setgroup can set group to empty string."""
        p = obj_peak.peak(mz=500.0, group='test')
        p.setgroup('')
        assert p.group == ''


# ============================================================================
# PROPERTY-BASED TESTS using Hypothesis
# ============================================================================

class TestPeakPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        mz=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        ai=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        base=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
    )
    def test_intensity_property(self, mz, ai, base):
        """Property: intensity = ai - base."""
        p = obj_peak.peak(mz=mz, ai=ai, base=base)
        assert p.intensity == pytest.approx(ai - base)

    @given(
        mz=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        fwhm=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    def test_resolution_property(self, mz, fwhm):
        """Property: resolution = mz / fwhm when fwhm is truthy."""
        p = obj_peak.peak(mz=mz, fwhm=fwhm)
        expected_resolution = mz / fwhm
        assert p.resolution == pytest.approx(expected_resolution)

    @given(
        mz=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        charge=st.integers(min_value=1, max_value=10)
    )
    def test_mass_consistency_property(self, mz, charge):
        """Property: mass() result is consistent across calls."""
        p = obj_peak.peak(mz=mz, charge=charge)
        first_call = p.mass()
        second_call = p.mass()
        assert first_call == second_call

    @given(
        charge=st.integers(min_value=-10, max_value=10)
    )
    def test_setcharge_affects_mass_buffer(self, charge):
        """Property: setcharge clears the mass buffer."""
        p = obj_peak.peak(mz=500.0, charge=1)
        p._mass = 999.0
        p.setcharge(charge)
        assert p._mass is None

    @given(
        mz_old=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        mz_new=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        fwhm=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    def test_setmz_updates_resolution_property(self, mz_old, mz_new, fwhm):
        """Property: setmz updates resolution correctly when fwhm is set."""
        p = obj_peak.peak(mz=mz_old, fwhm=fwhm)
        p.setmz(mz_new)
        expected_resolution = mz_new / fwhm
        assert p.resolution == pytest.approx(expected_resolution)


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================

class TestPeakEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_peak_with_very_small_mz(self):
        """Test peak with very small m/z value."""
        p = obj_peak.peak(mz=0.001, ai=100.0, base=10.0)
        assert p.mz == 0.001
        assert p.intensity == 90.0

    def test_peak_with_very_large_mz(self):
        """Test peak with very large m/z value."""
        p = obj_peak.peak(mz=100000.0, ai=100.0, base=10.0)
        assert p.mz == 100000.0

    def test_peak_with_very_small_fwhm(self):
        """Test peak with very small FWHM (high resolution)."""
        p = obj_peak.peak(mz=500.0, fwhm=0.001)
        assert p.resolution == pytest.approx(500000.0)

    def test_peak_with_very_large_fwhm(self):
        """Test peak with very large FWHM (low resolution)."""
        p = obj_peak.peak(mz=500.0, fwhm=100.0)
        assert p.resolution == pytest.approx(5.0)

    def test_peak_ai_equals_base(self):
        """Test peak where ai equals base (zero intensity)."""
        p = obj_peak.peak(mz=500.0, ai=50.0, base=50.0)
        assert p.intensity == 0.0

    def test_peak_ai_less_than_base(self):
        """Test peak where ai is less than base (negative intensity)."""
        p = obj_peak.peak(mz=500.0, ai=10.0, base=50.0)
        assert p.intensity == -40.0

    def test_peak_with_charge_zero(self):
        """Test peak with charge=0 (not None, special case)."""
        p = obj_peak.peak(mz=500.0, charge=0)
        assert p.charge == 0
        result = p.mass()
        # Should compute without error, even with charge=0
        assert result is not None

    def test_peak_with_negative_charge(self):
        """Test peak with negative charge."""
        p = obj_peak.peak(mz=500.0, charge=-2)
        result = p.mass()
        assert result is not None

    def test_peak_reset_multiple_times(self):
        """Test resetting peak multiple times."""
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0, fwhm=0.5, charge=1)
        p.reset()
        p.reset()
        p.reset()
        assert p.intensity == 90.0
        assert p.resolution == 1000.0
        assert p._mass is None

    def test_peak_string_group_name(self):
        """Test peak with group name."""
        p = obj_peak.peak(mz=500.0, group='isotope_cluster_1')
        assert p.group == 'isotope_cluster_1'

    def test_peak_isotope_patterns(self):
        """Test peak with different isotope patterns."""
        for isotope in ['M', 'M+1', 'M+2', '13C', '15N']:
            p = obj_peak.peak(mz=500.0, isotope=isotope)
            assert p.isotope == isotope


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPeakIntegration:
    """Integration tests combining multiple methods."""

    def test_peak_full_workflow(self):
        """Test a complete workflow of peak creation and modification."""
        # Create peak
        p = obj_peak.peak(mz=500.0, ai=100.0, base=10.0, charge=2, fwhm=0.5)

        # Verify initial state
        assert p.intensity == 90.0
        assert p.resolution == 1000.0
        assert p._mass is None

        # Get mass
        mass1 = p.mass()
        assert mass1 is not None
        assert p._mass == mass1

        # Change mz - should update resolution and clear mass
        p.setmz(600.0)
        assert p.resolution == 1200.0
        assert p._mass is None

        # Get mass again - should be recalculated
        mass2 = p.mass()
        assert mass2 is not None

        # Change charge - should clear mass
        p.setcharge(3)
        assert p._mass is None

        # Change fwhm - should update resolution
        p.setfwhm(1.0)
        assert p.resolution == 600.0

        # Reset
        p.setai(200.0)
        p.setbase(30.0)
        p.reset()
        assert p.intensity == 170.0
        assert p._mass is None

    def test_multiple_peaks_independent(self):
        """Test that multiple peak objects are independent."""
        p1 = obj_peak.peak(mz=500.0, ai=100.0, charge=1)
        p2 = obj_peak.peak(mz=600.0, ai=150.0, charge=2)

        p1._mass = 999.0
        p2._mass = 888.0

        assert p1._mass != p2._mass

        p1.setmz(550.0)
        assert p2.mz == 600.0

    def test_mass_caching_with_modifications(self):
        """Test mass caching behavior with various modifications."""
        p = obj_peak.peak(mz=500.0, charge=1)
        mass1 = p.mass()

        # Changing other parameters shouldn't clear mass
        p.setai(200.0)
        p.setbase(50.0)
        mass2 = p.mass()
        assert mass1 == mass2

        # Changing mz should clear mass
        p.setmz(600.0)
        assert p._mass is None

        # Changing charge should clear mass
        p.setcharge(2)
        assert p._mass is None

    def test_attributes_dict_storage(self):
        """Test that attributes are stored and retrieved correctly."""
        p = obj_peak.peak(
            mz=500.0,
            attr1='value1',
            attr2=42,
            attr3=3.14,
            nested={'key': 'value'}
        )
        assert p.attributes['attr1'] == 'value1'
        assert p.attributes['attr2'] == 42
        assert p.attributes['attr3'] == 3.14
        assert p.attributes['nested'] == {'key': 'value'}


# ============================================================================
# BOUNDARY VALUE TESTS
# ============================================================================

class TestPeakBoundaries:
    """Test boundary values and limits."""

    def test_resolution_with_equal_mz_and_fwhm(self):
        """Test resolution when mz equals fwhm."""
        p = obj_peak.peak(mz=1.0, fwhm=1.0)
        assert p.resolution == 1.0

    def test_resolution_with_mz_larger_than_fwhm(self):
        """Test resolution when mz is much larger than fwhm."""
        p = obj_peak.peak(mz=10000.0, fwhm=0.001)
        assert p.resolution == pytest.approx(10000000.0)

    def test_intensity_with_zero_values(self):
        """Test intensity calculation with zero ai and base."""
        p = obj_peak.peak(mz=500.0, ai=0.0, base=0.0)
        assert p.intensity == 0.0

    def test_float_precision_in_intensity(self):
        """Test that float precision is maintained in intensity."""
        p = obj_peak.peak(mz=500.0, ai=100.123456, base=10.654321)
        expected = 100.123456 - 10.654321
        assert p.intensity == pytest.approx(expected)

    def test_setfwhm_from_none_to_value(self):
        """Test setting fwhm from None to a value."""
        p = obj_peak.peak(mz=500.0, fwhm=None)
        assert p.resolution is None
        p.setfwhm(0.5)
        assert p.resolution == 1000.0

    def test_setfwhm_from_value_to_none(self):
        """Test setting fwhm from a value to None."""
        p = obj_peak.peak(mz=500.0, fwhm=0.5)
        assert p.resolution == 1000.0
        p.setfwhm(None)
        assert p.resolution is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
