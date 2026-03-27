import pytest
from hypothesis import given, strategies as st
import mspy.mod_basics
import mspy.obj_compound
import math

@pytest.mark.parametrize("measured, counted, unit, expected", [
    (1000.1, 1000.0, 'ppm', 100.0),
    (1000.1, 1000.0, 'Da', 0.1),
    (1000.1, 1000.0, '%', 0.01),
    (500.0, 1000.0, 'ppm', -500000.0),
    (1001.0, 1000.0, 'ppm', 1000.0),
    (1000.0, 1000.0, 'ppm', 0.0),
])
def test_delta_units(measured, counted, unit, expected):
    """Test delta function with different units."""
    result = mspy.mod_basics.delta(measured, counted, units=unit)
    assert result == pytest.approx(expected)

def test_delta_invalid_unit():
    """Test delta function with invalid unit."""
    with pytest.raises(ValueError):
        mspy.mod_basics.delta(1000.0, 1000.0, units='invalid')

@pytest.mark.parametrize("mass, rounding, expected", [
    (1.1, 'floor', 1.0),
    (1.9, 'floor', 1.0),
    (1.1, 'ceil', 2.0),
    (1.9, 'ceil', 2.0),
    (1.1, 'round', 1.0),
    (1.9, 'round', 2.0),
    (1.5, 'round', 2.0),
    (-1.1, 'floor', -2.0),
    (-1.9, 'floor', -2.0),
    (-1.1, 'ceil', -1.0),
    (-1.9, 'ceil', -1.0),
    (-1.1, 'round', -1.0),
    (-1.9, 'round', -2.0),
    (-1.5, 'round', -2.0), # Python 2.7 round(-1.5) is -2.0
])
def test_nominalmass_rounding(mass, rounding, expected):
    """Test nominalmass function with different rounding methods."""
    result = mspy.mod_basics.nominalmass(mass, rounding=rounding)
    assert result == expected

def test_nominalmass_invalid_rounding():
    """Test nominalmass function with invalid rounding method."""
    with pytest.raises(ValueError):
        mspy.mod_basics.nominalmass(1.5, rounding='invalid')

# Hypothesis test for delta (ppm) to ensure robustness
@given(
    measured=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    counted=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)
)
def test_delta_ppm_hypothesis(measured, counted):
    """Hypothesis test for delta ppm."""
    res = mspy.mod_basics.delta(measured, counted, units='ppm')
    expected = (measured - counted) / counted * 1000000
    assert res == pytest.approx(expected)

# Hypothesis test for nominalmass
@given(
    mass=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    rounding=st.sampled_from(['floor', 'ceil', 'round'])
)
def test_nominalmass_hypothesis(mass, rounding):
    """Hypothesis test for nominalmass."""
    res = mspy.mod_basics.nominalmass(mass, rounding=rounding)
    if rounding == 'floor':
        assert res == math.floor(mass)
    elif rounding == 'ceil':
        assert res == math.ceil(mass)
    elif rounding == 'round':
        assert res == round(mass)

def test_md_types():
    """Test md function with different types."""
    mass = 1000.12345
    # fraction
    assert mspy.mod_basics.md(mass, mdType='fraction') == pytest.approx(0.12345)
    
    # standard (floor)
    assert mspy.mod_basics.md(mass, mdType='standard', rounding='floor') == pytest.approx(0.12345)
    
    # standard (ceil)
    assert mspy.mod_basics.md(mass, mdType='standard', rounding='ceil') == pytest.approx(1000.12345 - 1001.0)
    
    # relative
    expected_rel = 1e6 * (1000.12345 - 1000.0) / 1000.12345
    assert mspy.mod_basics.md(mass, mdType='relative', rounding='floor') == pytest.approx(expected_rel)

def test_md_kendrick():
    """Test md function with Kendrick mass defect."""
    mass = 1000.12345
    # Kendrick with CH2 string
    ch2 = mspy.obj_compound.compound('CH2')
    kf = float(ch2.nominalmass()) / ch2.mass(0)
    expected_km = mspy.mod_basics.nominalmass(mass * kf, 'floor') - (mass * kf)
    assert mspy.mod_basics.md(mass, mdType='kendrick', kendrickFormula='CH2') == pytest.approx(expected_km)
    
    # Kendrick with compound object
    assert mspy.mod_basics.md(mass, mdType='kendrick', kendrickFormula=ch2) == pytest.approx(expected_km)

def test_md_invalid_type():
    """Test md function with invalid type."""
    with pytest.raises(ValueError):
        mspy.mod_basics.md(1000.0, mdType='invalid')

def test_mz_charge_zero():
    """Test mz function with charge zero."""
    mass = 1000.0
    assert mspy.mod_basics.mz(mass, 0) == mass
    
    mass_tuple = (1000.0, 1000.1)
    assert mspy.mod_basics.mz(mass_tuple, 0) == mass_tuple

def test_mz_current_charge():
    """Test mz function with non-zero current charge."""
    # Test currentCharge != 0 (e.g., [1000, 1000] with z=1 and agent='H' should give mass near 999).
    mass = (1000.0, 1000.0)
    z = 1
    agent = 'H'
    
    # neutral mass calculation for comparison
    h = mspy.obj_compound.compound(agent)
    h_mass = h.mass()
    agent_mass = (h_mass[0] - mspy.mod_basics.ELECTRON_MASS, h_mass[1] - mspy.mod_basics.ELECTRON_MASS)
    
    neutral_mo = mass[0] * abs(z) - agent_mass[0] * (z / 1.0)
    neutral_av = mass[1] * abs(z) - agent_mass[1] * (z / 1.0)
    
    # Verify mz(mass, 0, currentCharge=1) returns neutral mass
    res_neutral = mspy.mod_basics.mz(mass, 0, currentCharge=z, agentFormula=agent)
    assert res_neutral[0] == pytest.approx(neutral_mo)
    assert res_neutral[1] == pytest.approx(neutral_av)
    
    # Verify it's near 999
    assert neutral_mo == pytest.approx(998.9927, abs=1e-3)

def test_mz_electron():
    """Test mz function with electron as agent."""
    mass = 1000.0
    # agentFormula='e' uses ELECTRON_MASS directly
    expected = (1000.0 + mspy.mod_basics.ELECTRON_MASS * 1) / 1.0
    assert mspy.mod_basics.mz(1000.0, -1, agentFormula='e', agentCharge=-1) == pytest.approx(expected)

def test_mz_mass_types():
    """Test mz function with different mass input types."""
    mass_tuple = (1000.0, 1000.1)
    mass_list = [1000.0, 1000.1]
    
    h = mspy.obj_compound.compound('H')
    h_mass = h.mass()
    agent_mass = (h_mass[0] - mspy.mod_basics.ELECTRON_MASS, h_mass[1] - mspy.mod_basics.ELECTRON_MASS)
    
    expected_mo = (1000.0 + agent_mass[0]) / 1.0
    expected_av = (1000.1 + agent_mass[1]) / 1.0
    
    res_tuple = mspy.mod_basics.mz(mass_tuple, 1)
    assert res_tuple[0] == pytest.approx(expected_mo)
    assert res_tuple[1] == pytest.approx(expected_av)
    
    res_list = mspy.mod_basics.mz(mass_list, 1)
    assert res_list[0] == pytest.approx(expected_mo)
    assert res_list[1] == pytest.approx(expected_av)

def test_mz_agent_compound():
    """Test mz function with compound object as agent."""
    mass = 1000.0
    agent = mspy.obj_compound.compound('H')
    
    res1 = mspy.mod_basics.mz(mass, 1, agentFormula='H')
    res2 = mspy.mod_basics.mz(mass, 1, agentFormula=agent)
    assert res1 == pytest.approx(res2)

def test_mz_massType():
    """Test mz function with massType 0 and 1."""
    mass = 1000.0
    h = mspy.obj_compound.compound('H')
    h_mass = h.mass()
    agent_mass_mo = h_mass[0] - mspy.mod_basics.ELECTRON_MASS
    agent_mass_av = h_mass[1] - mspy.mod_basics.ELECTRON_MASS
    
    res0 = mspy.mod_basics.mz(mass, 1, massType=0)
    assert res0 == pytest.approx(mass + agent_mass_mo)
    
    res1 = mspy.mod_basics.mz(mass, 1, massType=1)
    assert res1 == pytest.approx(mass + agent_mass_av)

def test_mz_float_current_charge():
    """Test mz function with float mass and non-zero current charge."""
    mass = 1000.0
    z = 1
    # expected neutral mass
    h = mspy.obj_compound.compound('H')
    h_mass = h.mass()
    agent_mass_mo = h_mass[0] - mspy.mod_basics.ELECTRON_MASS
    
    expected_neutral = mass * abs(z) - agent_mass_mo * (z / 1.0)
    res = mspy.mod_basics.mz(mass, 0, currentCharge=z)
    assert res == pytest.approx(expected_neutral)

def test_rdbe():
    """Test rdbe function."""
    # Benzene C6H6: RDBE = 4.0
    assert mspy.mod_basics.rdbe("C6H6") == pytest.approx(4.0)
    
    # Cyclohexane C6H12: RDBE = 1.0
    assert mspy.mod_basics.rdbe("C6H12") == pytest.approx(1.0)
    
    # Water H2O: RDBE = 0.0
    assert mspy.mod_basics.rdbe("H2O") == pytest.approx(0.0)
    
    # Compound object
    c = mspy.obj_compound.compound("C6H6")
    assert mspy.mod_basics.rdbe(c) == pytest.approx(4.0)

def test_frules_hc():
    """Test frules HC rule."""
    # Pass: C6H6, ratioHC = 1.0 (range 0.1, 3.0)
    assert mspy.mod_basics.frules("C6H6", rules=['HC']) is True
    
    # Fail: ratioHC < 0.1
    assert mspy.mod_basics.frules("C20H1", rules=['HC']) is False
    
    # Fail: ratioHC > 3.0
    assert mspy.mod_basics.frules("CH4", rules=['HC']) is False

def test_frules_nopsc():
    """Test frules NOPSC rule."""
    # Default NOPSC=(4,3,2,3)
    # Pass
    assert mspy.mod_basics.frules("C10N2O2P2S2", rules=['NOPSC']) is True
    
    # Fail: ratioNC > 4
    assert mspy.mod_basics.frules("CN5", rules=['NOPSC']) is False
    
    # Fail: ratioOC > 3
    assert mspy.mod_basics.frules("CO4", rules=['NOPSC']) is False
    
    # Fail: ratioPC > 2
    assert mspy.mod_basics.frules("CP3", rules=['NOPSC']) is False
    
    # Fail: ratioSC > 3
    assert mspy.mod_basics.frules("CS4", rules=['NOPSC']) is False

def test_frules_nops_boundaries():
    """Test frules NOPS boundary rules."""
    # 1. countN > 1, countO > 1, countP > 1, countS > 1
    # Pass: satisfies all branches if they were to trigger
    assert mspy.mod_basics.frules("C10N2O2P2S2", rules=['NOPS']) is True
    # Fail N >= 10
    assert mspy.mod_basics.frules("C10N10O2P2S2", rules=['NOPS']) is False
    # Fail O >= 20
    assert mspy.mod_basics.frules("C10N2O20P2S2", rules=['NOPS']) is False
    # Fail P >= 4
    assert mspy.mod_basics.frules("C10N2O2P4S2", rules=['NOPS']) is False
    # Fail S >= 3
    assert mspy.mod_basics.frules("C10N2O2P2S3", rules=['NOPS']) is False

    # 2. countN > 3, countO > 3, countP > 3
    # Branch: countN > 3 and countO > 3 and countP > 3
    # Fail N >= 11
    assert mspy.mod_basics.frules("C10N11O4P4", rules=['NOPS']) is False
    # Fail O >= 22
    assert mspy.mod_basics.frules("C10N4O22P4", rules=['NOPS']) is False
    # Fail P >= 6
    assert mspy.mod_basics.frules("C10N4O4P6", rules=['NOPS']) is False

    # 3. countN > 1, countO > 1, countS > 1
    # Branch: countN > 1 and countO > 1 and countS > 1
    # Fail N >= 19
    assert mspy.mod_basics.frules("C10N19O2S2", rules=['NOPS']) is False
    # Fail O >= 14
    assert mspy.mod_basics.frules("C10N2O14S2", rules=['NOPS']) is False
    # Fail S >= 8
    assert mspy.mod_basics.frules("C10N2O2S8", rules=['NOPS']) is False

    # 4. countN > 1, countP > 1, countS > 1
    # Branch: countN > 1 and countP > 1 and countS > 1
    # Fail N >= 3
    assert mspy.mod_basics.frules("C10N3P2S2", rules=['NOPS']) is False
    # Fail P >= 3
    assert mspy.mod_basics.frules("C10N2P3S2", rules=['NOPS']) is False
    # Fail S >= 3
    assert mspy.mod_basics.frules("C10N2P2S3", rules=['NOPS']) is False

    # 5. countO > 1, countP > 1, countS > 1
    # Branch: countO > 1 and countP > 1 and countS > 1
    # Fail O >= 14
    assert mspy.mod_basics.frules("C10O14P2S2", rules=['NOPS']) is False
    # Fail P >= 3
    assert mspy.mod_basics.frules("C10O2P3S2", rules=['NOPS']) is False
    # Fail S >= 3
    assert mspy.mod_basics.frules("C10O2P2S3", rules=['NOPS']) is False

def test_frules_rdbe():
    """Test frules RDBE and RDBEInt rules."""
    # RDBE range (-1, 40)
    # Pass
    assert mspy.mod_basics.frules("C6H6", rules=['RDBE']) is True
    # Fail: RDBE > 40
    # C41H2 -> (41*2 - 2)/2 + 1 = 80/2 + 1 = 41.0
    assert mspy.mod_basics.frules("C41H2", rules=['RDBE']) is False
    # Fail: RDBE < -1 (not easy with normal elements, but let's try something weird if possible)
    # H has valence 1, so (1-2) = -1. H4 -> rdbe = -4/2 + 1 = -1.
    # H6 -> rdbe = -6/2 + 1 = -2.
    assert mspy.mod_basics.frules("H6", rules=['RDBE']) is False

    # RDBEInt
    # Pass: Benzene (4.0)
    assert mspy.mod_basics.frules("C6H6", rules=['RDBEInt']) is True
    # Fail: C6H7 (rdbe = (12-7)/2 + 1 = 3.5)
    assert mspy.mod_basics.frules("C6H7", rules=['RDBEInt']) is False

def test_frules_carbonless():
    """Test frules with carbon-less compounds."""
    # H2O: countC = 0.
    # Rules 'HC' and 'NOPSC' depend on countC.
    # NOPS rules do not.
    assert mspy.mod_basics.frules("H2O", rules=['HC', 'NOPSC', 'NOPS', 'RDBE', 'RDBEInt']) is True
    
    # Fail NOPS for H2O (not possible as it needs N, P, S too)
    # But RDBE range could fail
    assert mspy.mod_basics.frules("H2O", rules=['RDBE'], RDBE=(1, 10)) is False

def test_frules_toggling_rules():
    """Test toggling individual rules in frules."""
    # C6H7 fails RDBEInt but passes others
    assert mspy.mod_basics.frules("C6H7", rules=['HC', 'NOPSC', 'RDBE']) is True
    assert mspy.mod_basics.frules("C6H7", rules=['RDBEInt']) is False
    
    # CH4 fails HC (>3) but passes RDBEInt (rdbe=0)
    assert mspy.mod_basics.frules("CH4", rules=['RDBEInt']) is True
    assert mspy.mod_basics.frules("CH4", rules=['HC']) is False

def test_rdbe_duplicate_isotopes():
    """Test rdbe function with duplicate isotopes to trigger branch on line 196."""
    # C{12}C{13} -> symbols are 'C' and 'C'.
    # composition() returns {'C{12}': 1, 'C{13}': 1}
    # First item: symbol 'C' added to atoms.
    # Second item: symbol 'C' already in atoms, skipped.
    # rdbe = 1 + (2 * (4-2)) / 2 = 3.0
    assert mspy.mod_basics.rdbe("C{12}C{13}") == pytest.approx(3.0)

def test_rdbe_zero_valence():
    """Test rdbe function with zero valence element (Helium) to trigger branch on line 203."""
    # Helium has valence 0. rdbe = 1.0
    assert mspy.mod_basics.rdbe("He") == pytest.approx(1.0)

def test_frules_compound_instance():
    """Test frules with already instantiated compound object to trigger branch on line 222."""
    c = mspy.obj_compound.compound("C6H6")
    assert mspy.mod_basics.frules(c) is True

def test_frules_nops_branch_261():
    """Test frules NOPS rule with branch on line 261."""
    # countN > 3, countO > 3, countP > 3
    # but (countN >= 11 or countO >= 22 or countP >= 6) is False
    # Formula C10N4O4P4
    assert mspy.mod_basics.frules("C10N4O4P4", rules=['NOPS']) is True
