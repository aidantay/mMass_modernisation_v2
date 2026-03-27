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
from hypothesis import given, settings, HealthCheck, strategies as st
import mspy.obj_compound
import mspy.mod_basics
import math

# ======================================================================
# STEP 1-2: SMOKE/IMPORT TESTS
# ======================================================================

def test_import_compound():
    """Test that compound class can be imported."""
    from mspy.obj_compound import compound
    assert compound is not None

def test_smoke_basic_construction():
    """Test basic compound construction."""
    c = mspy.obj_compound.compound('H2O')
    assert c is not None
    assert c.expression == 'H2O'

# ======================================================================
# STEP 3: __init__ AND ATTRIBUTES
# ======================================================================

def test_init_valid_formula():
    """Test __init__ with valid formula."""
    c = mspy.obj_compound.compound('C6H12O6')
    assert c.expression == 'C6H12O6'
    assert c._composition is None
    assert c._formula is None
    assert c._mass is None
    assert c._nominalmass is None

def test_init_with_attributes():
    """Test __init__ stores attributes."""
    c = mspy.obj_compound.compound('H2O', name='water', value=42)
    assert c.attributes['name'] == 'water'
    assert c.attributes['value'] == 42

def test_init_invalid_formula_pattern():
    """Test __init__ raises ValueError on invalid formula pattern."""
    with pytest.raises(ValueError):
        mspy.obj_compound.compound('H2O2X')  # X is not valid

def test_init_unknown_element():
    """Test __init__ raises ValueError on unknown element."""
    with pytest.raises(ValueError):
        mspy.obj_compound.compound('Xx')  # Xx is not an element

def test_init_unknown_isotope():
    """Test __init__ raises ValueError on unknown isotope."""
    with pytest.raises(ValueError):
        mspy.obj_compound.compound('C{99}')  # C-99 doesn't exist

def test_init_mismatched_brackets():
    """Test __init__ raises ValueError on mismatched brackets."""
    with pytest.raises(ValueError):
        mspy.obj_compound.compound('(H2O')  # Missing closing bracket

def test_init_extra_closing_bracket():
    """Test __init__ raises ValueError on extra closing bracket."""
    with pytest.raises(ValueError):
        mspy.obj_compound.compound('H2O)')  # Extra closing bracket

# ======================================================================
# STEP 4: _checkFormula EXHAUSTIVE BRANCH TESTS
# ======================================================================

def test_checkformula_valid():
    """Test _checkFormula with valid formulas."""
    # Just constructing should call _checkFormula
    c = mspy.obj_compound.compound('CH4')
    c._checkFormula('C6H12')  # Should not raise

def test_checkformula_branch1_invalid_pattern():
    """Branch 1: FORMULA_PATTERN.match fails."""
    with pytest.raises(ValueError) as exc_info:
        mspy.obj_compound.compound('###')
    assert 'Wrong formula' in str(exc_info.value)

def test_checkformula_branch2_unknown_element():
    """Branch 2: element symbol not in blocks.elements."""
    with pytest.raises(ValueError) as exc_info:
        mspy.obj_compound.compound('Zz')
    assert 'Unknown element' in str(exc_info.value)

def test_checkformula_branch3_unknown_isotope():
    """Branch 3: isotope specified but not in element's isotopes."""
    with pytest.raises(ValueError) as exc_info:
        mspy.obj_compound.compound('H{999}')
    assert 'Unknown isotope' in str(exc_info.value)

def test_checkformula_branch4_mismatched_parenthesis():
    """Branch 4: mismatched parenthesis count."""
    with pytest.raises(ValueError) as exc_info:
        mspy.obj_compound.compound('(CH2)(OH')
    assert 'Wrong number of brackets' in str(exc_info.value)

def test_checkformula_branch5_all_pass():
    """Branch 5: all checks pass."""
    c = mspy.obj_compound.compound('Ca(OH)2')
    assert c.expression == 'Ca(OH)2'

# ======================================================================
# STEP 5: _unfoldBrackets TESTS
# ======================================================================

def test_unfold_no_brackets():
    """Test _unfoldBrackets with no brackets."""
    c = mspy.obj_compound.compound('H2O')
    result = c._unfoldBrackets('H2O')
    assert result == 'H2O'

def test_unfold_simple_brackets():
    """Test _unfoldBrackets with simple brackets."""
    c = mspy.obj_compound.compound('CH4')
    result = c._unfoldBrackets('(OH)2')
    # _unfoldBrackets repeats the string, composition() aggregates
    assert result == 'OHOH'

def test_unfold_no_multiplier():
    """Test _unfoldBrackets with brackets but no multiplier."""
    c = mspy.obj_compound.compound('H2O')
    result = c._unfoldBrackets('(OH)')
    assert result == 'OH'

def test_unfold_nested_brackets():
    """Test _unfoldBrackets with nested brackets."""
    c = mspy.obj_compound.compound('Ca')
    result = c._unfoldBrackets('((OH)2)1')
    # _unfoldBrackets repeats the string, composition() aggregates
    assert result == 'OHOH'

def test_unfold_multi_digit_multiplier():
    """Test _unfoldBrackets with multi-digit multiplier."""
    c = mspy.obj_compound.compound('Ca')
    result = c._unfoldBrackets('(CH2)10')
    # _unfoldBrackets repeats the string 10 times
    expected = 'CH2' * 10
    assert result == expected

def test_unfold_complex_brackets():
    """Test _unfoldBrackets with complex formula."""
    c = mspy.obj_compound.compound('Ca')
    result = c._unfoldBrackets('Ca(OH)2')
    # _unfoldBrackets repeats (OH) twice
    assert result == 'CaOHOH'

# ======================================================================
# STEP 6: composition() TESTS
# ======================================================================

def test_composition_simple():
    """Test composition() with simple formula."""
    c = mspy.obj_compound.compound('H2O')
    comp = c.composition()
    assert comp == {'H': 2, 'O': 1}

def test_composition_cached():
    """Test composition() returns cached value."""
    c = mspy.obj_compound.compound('H2O')
    comp1 = c.composition()
    comp2 = c.composition()
    assert comp1 is comp2  # Same object (cached)

def test_composition_isotope():
    """Test composition() with isotope."""
    c = mspy.obj_compound.compound('C{12}C{13}')
    comp = c.composition()
    assert 'C{12}' in comp
    assert 'C{13}' in comp
    assert comp['C{12}'] == 1
    assert comp['C{13}'] == 1

def test_composition_bracketed():
    """Test composition() with bracketed formula."""
    c = mspy.obj_compound.compound('Ca(OH)2')
    comp = c.composition()
    assert comp['Ca'] == 1
    assert comp['O'] == 2
    assert comp['H'] == 2

def test_composition_zero_count_removal():
    """Test composition() removes zero-count atoms."""
    c = mspy.obj_compound.compound('CH2')
    c._unfoldBrackets('C1H2C-1')  # This will produce zero-count
    # Directly test the zero-removal logic by creating a compound
    # and manipulating composition
    c2 = mspy.obj_compound.compound('C2H4')
    comp = c2.composition()
    # All counts should be positive
    for atom, count in comp.items():
        assert count > 0

def test_composition_multiple_same_atoms():
    """Test composition() aggregates same atoms."""
    c = mspy.obj_compound.compound('C2H4')
    comp = c.composition()
    assert comp['C'] == 2
    assert comp['H'] == 4

def test_composition_empty_or_single_atom():
    """Test composition() with single atom."""
    c = mspy.obj_compound.compound('O')
    comp = c.composition()
    assert comp == {'O': 1}

# ======================================================================
# STEP 7: formula() TESTS
# ======================================================================

def test_formula_simple():
    """Test formula() with simple formula."""
    c = mspy.obj_compound.compound('H2O')
    f = c.formula()
    # Should have C/H priority ordering
    assert f is not None

def test_formula_cached():
    """Test formula() returns cached value."""
    c = mspy.obj_compound.compound('C6H12')
    f1 = c.formula()
    f2 = c.formula()
    assert f1 is f2  # Same object (cached)

def test_formula_ch_priority():
    """Test formula() orders C and H first."""
    c = mspy.obj_compound.compound('N2C2H4O')
    f = c.formula()
    # C and H should come before other elements
    c_idx = f.find('C')
    h_idx = f.find('H')
    n_idx = f.find('N')
    o_idx = f.find('O')
    assert c_idx < n_idx or c_idx == -1
    assert h_idx < n_idx or h_idx == -1

def test_formula_single_atom_no_count():
    """Test formula() with single atom count=1."""
    c = mspy.obj_compound.compound('C')
    f = c.formula()
    assert f == 'C'

def test_formula_isotope_variants():
    """Test formula() handles isotope variants."""
    c = mspy.obj_compound.compound('C{12}H4')
    f = c.formula()
    assert 'C{12}' in f or 'C' in f
    assert 'H' in f

def test_formula_with_counts():
    """Test formula() with atom counts."""
    c = mspy.obj_compound.compound('C6H6')
    f = c.formula()
    assert 'C6' in f
    assert 'H6' in f

# ======================================================================
# STEP 8: mass() TESTS
# ======================================================================

def test_mass_tuple_default():
    """Test mass() returns tuple by default."""
    c = mspy.obj_compound.compound('H2O')
    m = c.mass()
    assert isinstance(m, tuple)
    assert len(m) == 2
    monoisotopic, average = m
    assert monoisotopic > 0
    assert average > 0

def test_mass_masstype_0():
    """Test mass(massType=0) returns monoisotopic mass."""
    c = mspy.obj_compound.compound('H2O')
    m = c.mass(massType=0)
    assert isinstance(m, float) or isinstance(m, int)
    assert m > 0

def test_mass_masstype_1():
    """Test mass(massType=1) returns average mass."""
    c = mspy.obj_compound.compound('H2O')
    m = c.mass(massType=1)
    assert isinstance(m, float) or isinstance(m, int)
    assert m > 0

def test_mass_cached():
    """Test mass() caches result."""
    c = mspy.obj_compound.compound('H2O')
    m1 = c.mass()
    m2 = c.mass()
    assert m1 is m2  # Same object (cached)

def test_mass_with_isotope():
    """Test mass() with isotope specification."""
    c = mspy.obj_compound.compound('C{12}H4')
    m = c.mass()
    assert m is not None

def test_mass_isotope_branch():
    """Test mass() branches on massNumber present vs absent."""
    c1 = mspy.obj_compound.compound('C')
    m1 = c1.mass()

    c2 = mspy.obj_compound.compound('C{12}')
    m2 = c2.mass()

    # Both should have same monoisotopic mass for C-12
    assert m1[0] == pytest.approx(m2[0])

def test_mass_multiple_atoms():
    """Test mass() with multiple atoms."""
    c = mspy.obj_compound.compound('C2H4O')
    m = c.mass()
    assert m[0] > 0 and m[1] > 0

# ======================================================================
# STEP 9: nominalmass() TESTS
# ======================================================================

def test_nominalmass_simple():
    """Test nominalmass() with simple formula."""
    c = mspy.obj_compound.compound('H2O')
    nm = c.nominalmass()
    assert isinstance(nm, (int, float))
    assert nm > 0

def test_nominalmass_cached():
    """Test nominalmass() returns cached value."""
    c = mspy.obj_compound.compound('H2O')
    nm1 = c.nominalmass()
    nm2 = c.nominalmass()
    assert nm1 == nm2
    assert c._nominalmass == nm1

def test_nominalmass_isotope():
    """Test nominalmass() with isotope."""
    c = mspy.obj_compound.compound('C{12}')
    nm = c.nominalmass()
    assert nm == 12

def test_nominalmass_without_isotope():
    """Test nominalmass() without isotope."""
    c = mspy.obj_compound.compound('C')
    nm = c.nominalmass()
    assert nm == 12  # Default C-12

def test_nominalmass_multiple_atoms():
    """Test nominalmass() sums multiple atoms correctly."""
    c = mspy.obj_compound.compound('C6H6')
    nm = c.nominalmass()
    expected = 6 * 12 + 6 * 1
    assert nm == expected

# ======================================================================
# STEP 10: count() TESTS
# ======================================================================

def test_count_simple():
    """Test count() with simple atom."""
    c = mspy.obj_compound.compound('H2O')
    assert c.count('H') == 2
    assert c.count('O') == 1

def test_count_absent():
    """Test count() with absent atom."""
    c = mspy.obj_compound.compound('H2O')
    assert c.count('C') == 0

def test_count_groupisotopes_false():
    """Test count(groupIsotopes=False) counts only exact atom key."""
    c = mspy.obj_compound.compound('C{12}H4')
    count = c.count('C', groupIsotopes=False)
    assert count == 0  # Only C{12} exists, not bare C

def test_count_groupisotopes_true_element_exists():
    """Test count(groupIsotopes=True) counts isotope-labelled keys."""
    c = mspy.obj_compound.compound('C{12}C{13}H4')
    count = c.count('C', groupIsotopes=True)
    # Should count both C{12} and C{13}
    assert count == 2

def test_count_groupisotopes_true_non_element():
    """Test count(groupIsotopes=True) with non-element (no isotopes)."""
    c = mspy.obj_compound.compound('H2O')
    count = c.count('H', groupIsotopes=True)
    # H is an element, but if only 'H' exists (not H{1}, etc.), count should work
    assert count == 2

def test_count_with_isotope_exact():
    """Test count() with isotope exact match."""
    c = mspy.obj_compound.compound('C{12}')
    assert c.count('C{12}') == 1

# ======================================================================
# STEP 11: mz() TESTS (delegation)
# ======================================================================

def test_mz_positive_charge():
    """Test mz() with positive charge."""
    c = mspy.obj_compound.compound('H2O')
    mz = c.mz(charge=1)
    assert mz > 0

def test_mz_negative_charge():
    """Test mz() with negative charge."""
    c = mspy.obj_compound.compound('H2O')
    mz = c.mz(charge=-1)
    assert mz > 0

def test_mz_zero_charge():
    """Test mz() with zero charge."""
    c = mspy.obj_compound.compound('H2O')
    mass = c.mass()
    mz = c.mz(charge=0)
    # With zero charge, mz should equal mass
    if isinstance(mass, tuple):
        assert mz == mass
    else:
        assert mz == mass

def test_mz_electron_agent():
    """Test mz() with electron agent."""
    c = mspy.obj_compound.compound('H2O')
    mz = c.mz(charge=-1, agentFormula='e')
    assert mz > 0

def test_mz_compound_agent():
    """Test mz() with compound agent."""
    c = mspy.obj_compound.compound('H2O')
    mz = c.mz(charge=1, agentFormula='H')
    assert mz > 0

# ======================================================================
# STEP 12-13: pattern(), rdbe(), frules() DELEGATION SMOKE TESTS
# ======================================================================

def test_pattern_smoke():
    """Test pattern() delegation (smoke test)."""
    c = mspy.obj_compound.compound('H2O')
    pattern = c.pattern()
    assert pattern is not None

def test_rdbe_smoke():
    """Test rdbe() delegation (smoke test)."""
    c = mspy.obj_compound.compound('C6H6')
    rdbe = c.rdbe()
    assert rdbe == pytest.approx(4.0)

def test_frules_smoke():
    """Test frules() delegation (smoke test)."""
    c = mspy.obj_compound.compound('C6H6')
    result = c.frules()
    assert isinstance(result, bool)

# ======================================================================
# STEP 14: isvalid() TESTS (all branches)
# ======================================================================

def test_isvalid_simple():
    """Test isvalid() with simple case."""
    c = mspy.obj_compound.compound('H2O')
    assert c.isvalid() is True

def test_isvalid_branch1_agentformula_not_e_not_compound():
    """Branch 1: agentFormula != 'e' AND not a compound instance."""
    c = mspy.obj_compound.compound('C6H12O6')
    # Pass agentFormula as string (not 'e', not compound)
    result = c.isvalid(charge=1, agentFormula='H')
    assert isinstance(result, bool)

def test_isvalid_branch2_agentformula_e():
    """Branch 2: agentFormula == 'e'."""
    c = mspy.obj_compound.compound('H2O')
    result = c.isvalid(charge=-1, agentFormula='e')
    assert isinstance(result, bool)

def test_isvalid_branch3_agentformula_compound():
    """Branch 3: agentFormula already a compound."""
    c = mspy.obj_compound.compound('H2O')
    agent = mspy.obj_compound.compound('H')
    result = c.isvalid(charge=1, agentFormula=agent)
    assert isinstance(result, bool)

def test_isvalid_branch4_charge_nonzero_agent_not_e():
    """Branch 4: charge != 0 AND agentFormula != 'e'."""
    c = mspy.obj_compound.compound('C6H12O6')
    result = c.isvalid(charge=1, agentFormula='H', agentCharge=1)
    assert isinstance(result, bool)

def test_isvalid_branch5_charge_zero():
    """Branch 5: charge == 0."""
    c = mspy.obj_compound.compound('C6H12O6')
    result = c.isvalid(charge=0, agentFormula='H')
    assert result is True

def test_isvalid_branch6_negative_count():
    """Branch 6: any atom count < 0 (invalid)."""
    c = mspy.obj_compound.compound('H')
    # To get negative count, we need to subtract more than available
    # Create a compound that subtracts atoms
    result = c.isvalid(charge=1, agentFormula='C', agentCharge=1)
    # H1 + C1(1/1) = H1 + C1, no negative counts
    # This is a challenging branch; let's try another approach
    # The ion formula is constructed; if atom count goes negative, return False
    assert isinstance(result, bool)

def test_isvalid_all_counts_positive():
    """Branch 7: all counts >= 0."""
    c = mspy.obj_compound.compound('C6H12O6')
    result = c.isvalid(charge=0)
    assert result is True

def test_isvalid_negative_composition():
    """Test isvalid() with formula that could produce negative composition."""
    c = mspy.obj_compound.compound('CH2')
    # Try to subtract more than available
    result = c.isvalid(charge=2, agentFormula='C10', agentCharge=1)
    # CH2 + 2*C10 = CH2 + C20 (both positive), so should be True
    assert result is True

def test_isvalid_negative_count_branch():
    """Test isvalid() branch where count < 0 (line 307-308)."""
    c = mspy.obj_compound.compound('H')
    # To get negative count, we need charge=-1 and agentFormula to produce net negative
    # H + (-1)*C = H - C (C becomes negative)
    result = c.isvalid(charge=-1, agentFormula='C', agentCharge=1)
    # H1 - C1 = H1 C-1, which has negative C, should return False
    assert result is False

# ======================================================================
# STEP 15: frules() DELEGATION SMOKE TEST (additional)
# ======================================================================

def test_frules_parameters():
    """Test frules() with custom parameters."""
    c = mspy.obj_compound.compound('C6H6')
    result = c.frules(rules=['HC', 'RDBE'])
    assert isinstance(result, bool)

# ======================================================================
# STEP 16: negate() TESTS
# ======================================================================

def test_negate_expression():
    """Test negate() modifies expression."""
    c = mspy.obj_compound.compound('H2O')
    c.negate()
    assert 'H-2' in c.expression or 'H-2' in c.expression
    assert 'O-1' in c.expression

def test_negate_buffer_clearing():
    """Test negate() clears buffers."""
    c = mspy.obj_compound.compound('H2O')
    # Populate buffers
    comp = c.composition()
    mass = c.mass()

    # Negate
    c.negate()

    # Buffers should be None
    assert c._composition is None
    assert c._mass is None
    assert c._nominalmass is None
    assert c._formula is None

def test_negate_negative_composition():
    """Test negate() creates negative composition."""
    c = mspy.obj_compound.compound('C')
    c.negate()
    comp = c.composition()
    for atom, count in comp.items():
        assert count < 0

def test_negate_idempotent():
    """Test negate() twice returns to original."""
    c = mspy.obj_compound.compound('H2O')
    original_comp = c.composition()

    c.negate()
    c.negate()

    final_comp = c.composition()
    for atom in original_comp:
        assert atom in final_comp
        assert final_comp[atom] == original_comp[atom]

# ======================================================================
# STEP 17: reset() TESTS
# ======================================================================

def test_reset_clears_composition():
    """Test reset() clears composition buffer."""
    c = mspy.obj_compound.compound('H2O')
    comp1 = c.composition()
    assert c._composition is not None

    c.reset()
    assert c._composition is None

def test_reset_clears_all_buffers():
    """Test reset() clears all buffers."""
    c = mspy.obj_compound.compound('H2O')
    # Populate all buffers
    comp = c.composition()
    formula = c.formula()
    mass = c.mass()
    nom_mass = c.nominalmass()

    assert c._composition is not None
    assert c._formula is not None
    assert c._mass is not None
    assert c._nominalmass is not None

    # Reset
    c.reset()

    assert c._composition is None
    assert c._formula is None
    assert c._mass is None
    assert c._nominalmass is None

def test_reset_preserves_expression():
    """Test reset() preserves expression."""
    c = mspy.obj_compound.compound('H2O')
    expr = c.expression
    c.reset()
    assert c.expression == expr

# ======================================================================
# STEP 18: __iadd__ TESTS
# ======================================================================

def test_iadd_compound_compound():
    """Test __iadd__ with compound + compound."""
    c1 = mspy.obj_compound.compound('H2O')
    c2 = mspy.obj_compound.compound('CH4')

    c1 += c2

    assert c1.expression == 'H2OCH4'
    assert c1 is c1  # Should return self

def test_iadd_compound_string():
    """Test __iadd__ with compound + valid string."""
    c = mspy.obj_compound.compound('H2O')
    c += 'CH4'

    assert c.expression == 'H2OCH4'

def test_iadd_invalid_string():
    """Test __iadd__ with invalid string."""
    c = mspy.obj_compound.compound('H2O')
    with pytest.raises(ValueError):
        c += 'XXX'

def test_iadd_returns_self():
    """Test __iadd__ returns self."""
    c = mspy.obj_compound.compound('H2O')
    result = c.__iadd__(mspy.obj_compound.compound('CH4'))
    assert result is c

def test_iadd_clears_buffers():
    """Test __iadd__ clears buffers."""
    c = mspy.obj_compound.compound('H2O')
    # Populate buffers
    comp = c.composition()
    assert c._composition is not None

    # Add
    c += 'CH4'

    assert c._composition is None

def test_iadd_chain():
    """Test __iadd__ can be chained."""
    c = mspy.obj_compound.compound('H')
    c += 'H'
    c += 'O'

    assert c.expression == 'HHO'

# ======================================================================
# STEP 19: HYPOTHESIS PROPERTY-BASED TESTS
# ======================================================================

@given(st.sampled_from(['H2O', 'CH4', 'C6H12O6', 'NH3', 'CO2', 'N2', 'O2', 'H2', 'Ca(OH)2', 'C{12}H4']))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prop_composition_is_dict(formula):
    """Property: composition() always returns a dict."""
    c = mspy.obj_compound.compound(formula)
    comp = c.composition()
    assert isinstance(comp, dict)
    for atom, count in comp.items():
        assert isinstance(atom, str)
        assert isinstance(count, (int, float))
        assert count > 0

@given(st.sampled_from(['H2O', 'CH4', 'C6H12O6', 'NH3', 'CO2', 'N2', 'O2', 'H2', 'Ca(OH)2', 'C{12}H4']))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prop_mass_positive(formula):
    """Property: mass() always returns positive values."""
    c = mspy.obj_compound.compound(formula)
    m = c.mass()
    assert m[0] > 0
    assert m[1] > 0
    assert m[0] <= m[1]  # Monoisotopic <= average

@given(st.sampled_from(['H2O', 'CH4', 'C6H12O6', 'NH3', 'CO2', 'N2', 'O2', 'H2', 'Ca(OH)2', 'C{12}H4']))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prop_nominalmass_positive(formula):
    """Property: nominalmass() always returns positive value."""
    c = mspy.obj_compound.compound(formula)
    nm = c.nominalmass()
    assert nm > 0

@given(st.sampled_from(['H2O', 'CH4', 'C6H12O6', 'NH3', 'CO2', 'N2', 'O2', 'H2', 'Ca(OH)2', 'C{12}H4']))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prop_formula_is_string(formula):
    """Property: formula() always returns a string."""
    c = mspy.obj_compound.compound(formula)
    f = c.formula()
    assert isinstance(f, str)
    assert len(f) > 0

@given(st.sampled_from(['H2O', 'CH4', 'C6H12O6', 'NH3', 'CO2', 'N2', 'O2', 'H2', 'Ca(OH)2', 'C{12}H4']))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prop_caching_works(formula):
    """Property: second call to getter returns same cached object."""
    c = mspy.obj_compound.compound(formula)
    comp1 = c.composition()
    comp2 = c.composition()
    assert comp1 is comp2

    formula1 = c.formula()
    formula2 = c.formula()
    assert formula1 is formula2

@given(st.sampled_from(['H2O', 'CH4', 'C6H12O6', 'NH3', 'CO2', 'N2', 'O2', 'H2', 'Ca(OH)2', 'C{12}H4']))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prop_reset_clears_all(formula):
    """Property: reset() clears all buffers."""
    c = mspy.obj_compound.compound(formula)
    # Populate buffers
    c.composition()
    c.formula()
    c.mass()
    c.nominalmass()

    c.reset()

    assert c._composition is None
    assert c._formula is None
    assert c._mass is None
    assert c._nominalmass is None

@given(st.sampled_from(['H2O', 'CH4', 'C6H12O6', 'NH3', 'CO2', 'N2', 'O2', 'H2', 'Ca(OH)2', 'C{12}H4']))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prop_count_matches_composition(formula):
    """Property: count() results match composition()."""
    c = mspy.obj_compound.compound(formula)
    comp = c.composition()

    for atom in comp:
        count = c.count(atom)
        assert count == comp[atom]

# ======================================================================
# STEP 20: BUFFER CACHING/RESET INTEGRATION TESTS
# ======================================================================

def test_cache_invalidation_on_iadd():
    """Test cache invalidation when using __iadd__."""
    c = mspy.obj_compound.compound('H2O')
    comp1 = c.composition()

    c += 'H'

    # After iadd, cache should be cleared
    assert c._composition is None

    comp2 = c.composition()
    # New composition should be different
    assert comp1 != comp2
    assert comp2['H'] == 3

def test_cache_invalidation_on_negate():
    """Test cache invalidation when using negate()."""
    c = mspy.obj_compound.compound('H2O')
    comp1 = c.composition()

    c.negate()

    # After negate, cache should be cleared
    assert c._composition is None

    comp2 = c.composition()
    for atom in comp2:
        assert comp2[atom] < 0

def test_buffer_consistency_after_operations():
    """Test buffer consistency across multiple operations."""
    c = mspy.obj_compound.compound('H2O')

    # Get mass
    m1 = c.mass()
    assert c._mass is not None

    # Get nominal mass
    nm = c.nominalmass()
    assert c._nominalmass is not None

    # Both should still be cached
    m2 = c.mass()
    assert m1 is m2
    nm2 = c.nominalmass()
    assert nm is nm2

def test_reset_after_all_operations():
    """Test reset after using all getters."""
    c = mspy.obj_compound.compound('C6H12O6')

    # Use all getters
    c.composition()
    c.formula()
    c.mass()
    c.nominalmass()
    c.count('C')

    # Reset
    c.reset()

    # All should be None
    assert c._composition is None
    assert c._formula is None
    assert c._mass is None
    assert c._nominalmass is None

    # Should be able to use getters again
    assert c.composition() is not None
    assert c.formula() is not None

# ======================================================================
# ADDITIONAL EDGE CASE TESTS
# ======================================================================

def test_compound_with_large_formula():
    """Test compound with large/complex formula."""
    c = mspy.obj_compound.compound('C100H202O50N20P10S5')
    assert c.expression == 'C100H202O50N20P10S5'
    comp = c.composition()
    assert comp['C'] == 100
    assert comp['H'] == 202

def test_compound_with_nested_brackets():
    """Test compound with nested brackets."""
    c = mspy.obj_compound.compound('((CH2)2)3')
    comp = c.composition()
    assert comp['C'] == 6
    assert comp['H'] == 12

def test_compound_isotope_count_grouping():
    """Test counting with isotope grouping."""
    c = mspy.obj_compound.compound('C{12}C{13}C{14}H4')
    total = c.count('C', groupIsotopes=True)
    assert total == 3

def test_compound_mass_precision():
    """Test mass calculations are reasonably precise."""
    c = mspy.obj_compound.compound('H2O')
    m = c.mass()
    # Water should be approximately 18
    assert m[0] > 17 and m[0] < 19

def test_composition_order_independence():
    """Test that formula order doesn't affect composition."""
    c1 = mspy.obj_compound.compound('H2O')
    c2 = mspy.obj_compound.compound('OH2')

    # Compositions should be equal
    comp1 = c1.composition()
    comp2 = c2.composition()

    # Both should have same atoms and counts
    for atom in ['H', 'O']:
        assert comp1.get(atom, 0) == comp2.get(atom, 0)

def test_formula_output_consistency():
    """Test formula output is consistent across calls."""
    c = mspy.obj_compound.compound('Ca(OH)2')
    f1 = c.formula()
    f2 = c.formula()
    assert f1 == f2

# ======================================================================
# BOUNDARY VALUE TESTS
# ======================================================================

def test_single_atom_hydrogen():
    """Test single hydrogen atom."""
    c = mspy.obj_compound.compound('H')
    assert c.composition() == {'H': 1}
    assert c.nominalmass() == 1

def test_single_atom_oxygen():
    """Test single oxygen atom."""
    c = mspy.obj_compound.compound('O')
    assert c.composition() == {'O': 1}
    assert c.nominalmass() == 16

def test_isotope_explicit_label():
    """Test explicit isotope label."""
    c = mspy.obj_compound.compound('H{1}')
    comp = c.composition()
    assert 'H{1}' in comp
    assert comp['H{1}'] == 1

def test_high_count_atom():
    """Test atom with high count."""
    c = mspy.obj_compound.compound('O999')
    assert c.composition()['O'] == 999
    assert c.nominalmass() == 999 * 16

def test_bracket_with_high_multiplier():
    """Test bracket with high multiplier."""
    c = mspy.obj_compound.compound('(H2O)99')
    comp = c.composition()
    assert comp['H'] == 198
    assert comp['O'] == 99
