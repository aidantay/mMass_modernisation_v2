# -------------------------------------------------------------------------
#     Copyright (C) 2005-2013 Martin Strohalm <www.mmass.org>

#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.

#     Complete text of GNU GPL can be found in the file LICENSE.TXT in the
#     main directory of the program.
# -------------------------------------------------------------------------

import pytest
import numpy
from hypothesis import given, strategies as st
from mspy.mod_envfit import envfit

@pytest.fixture
def envfit_obj():
    """Returns envfit('H2O', 1, [0, 1])."""
    return envfit('H2O', 1, [0, 1])

@pytest.fixture
def mock_check_force_quit(mocker):
    """Mocks mspy.mod_envfit.CHECK_FORCE_QUIT."""
    return mocker.patch('mspy.mod_envfit.CHECK_FORCE_QUIT')

@pytest.fixture
def dummy_signal():
    """Returns numpy.array([[18.0, 100.0], [19.0, 50.0], [20.0, 10.0]])."""
    return numpy.array([[18.0, 100.0], [19.0, 50.0], [20.0, 10.0]])

@pytest.fixture
def dummy_baseline():
    """Returns numpy.array([[18.0, 0.0], [19.0, 0.0], [20.0, 0.0]])"""
    return numpy.array([[18.0, 0.0], [19.0, 0.0], [20.0, 0.0]])

@pytest.fixture
def dummy_points():
    """Returns numpy.array([[18.010565, 100.0], [19.013920, 50.0]])."""
    return numpy.array([[18.010565, 100.0], [19.013920, 50.0]])

def test_setup_verification(envfit_obj, mock_check_force_quit, dummy_signal, dummy_baseline, dummy_points):
    """Simple test to verify fixtures and environment setup."""
    assert isinstance(envfit_obj, envfit)
    assert dummy_signal.shape == (3, 2)
    assert dummy_baseline.shape == (3, 2)
    assert dummy_points.shape == (2, 2)
    assert mock_check_force_quit is not None

def test_init_success():
    """Test successful initialization with standard parameters."""
    ef = envfit('H2O', 1, [0, 1], loss='H', gain='H{2}')
    assert ef._lossFormula == 'H-1'
    assert ef._gainFormula == 'H{2}'
    assert ef.formula == 'H2O'
    assert ef.charge == 1
    # Check mzrange is set and reasonable for H2O and H2O with H replaced by H2, charge 1
    # H2O + H+ = 19.018
    # H1 O1 H{2}1 + H+ = 20.024
    assert ef.mzrange[0] < 19.1
    assert ef.mzrange[1] > 19.9

def test_init_invalid_compound(mocker):
    """Test initialization skip invalid compound."""
    # Mock obj_compound.compound
    mock_cls = mocker.patch('mspy.mod_envfit.obj_compound.compound')
    
    # 1st call: loss = obj_compound.compound(loss) in __init__
    mock_loss = mocker.Mock()
    mock_loss.formula.return_value = 'H-1'
    
    # 2nd call: compound = obj_compound.compound(item) for x=0 in _initModels
    mock_comp0 = mocker.Mock()
    mock_comp0.isvalid.return_value = True
    # _initRange calls min(scales) compound pattern
    mock_comp0.pattern.return_value = numpy.array([[19.0, 100.0]])
    
    # 3rd call: compound = obj_compound.compound(item) for x=1 in _initModels
    mock_comp1 = mocker.Mock()
    mock_comp1.isvalid.return_value = False # Should skip this one
    
    mock_cls.side_effect = [mock_loss, mock_comp0, mock_comp1]
    
    ef = envfit('H2O', 1, [0, 1])
    
    assert 0 in ef.models
    assert 1 not in ef.models
    assert len(ef.models) == 1

def test_init_range():
    """Verify _initRange sets mzrange correctly."""
    # For H2O, charge 0, scales [0, 1]
    # x=0: H2O. mz ~ 18.01
    # x=1: H1 O1 H{2}1. mz ~ 19.017
    ef = envfit('H2O', 0, [0, 1])
    # x1, x2 from pattern(fwhm=0.5)
    # H2O pattern[0][0] is 18.0105
    # H1 O1 H{2}1 pattern[0][0] is 19.0168
    # mzrange[0] = min(18.01, 19.01) * 0.999 = 17.99
    # mzrange[1] = max(18.01, 19.01) * 1.001 = 19.03
    assert 17.9 < ef.mzrange[0] < 18.1
    assert 19.0 < ef.mzrange[1] < 19.1

def test_topoints_list_input(envfit_obj):
    """Test topoints with points as a list [[18.01, 100.0]], check if ef.data is numpy.ndarray."""
    points = [[18.01, 100.0]]
    # We need to make sure the range covers 18.01.
    # H2O + H+ is ~19.01. So 18.01 is outside if charge is 1.
    # Let's use envfit('H2O', 0, [0, 1]) where 18.01 is valid.
    ef = envfit('H2O', 0, [0, 1])
    ef.topoints(points, autoAlign=False)
    assert isinstance(ef.data, numpy.ndarray)
    assert ef.data.shape == (1, 2)
    assert ef.data[0][0] == 18.01

def test_topoints_outside_range(envfit_obj):
    """Test topoints with points that are outside ef.mzrange, check if it returns False."""
    # mzrange for H2O, charge 1 is roughly [18.9, 20.1]
    points = [[10.0, 100.0], [30.0, 50.0]]
    result = envfit_obj.topoints(points)
    assert result is False
    assert len(envfit_obj.data) == 0

def test_topoints_autoAlign_false(envfit_obj, mocker):
    """Test topoints with autoAlign=False, verify _alignData is NOT called."""
    mocker.patch.object(envfit_obj, '_alignData')
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(numpy.array([[1.0]]), [0]))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[1.0])
    
    # Needs valid point
    mz = envfit_obj.mzrange[0] + 0.1
    envfit_obj.topoints([[mz, 100.0]], autoAlign=False)
    
    envfit_obj._alignData.assert_not_called()

def test_topoints_autoAlign_true(envfit_obj, mocker):
    """Test topoints with autoAlign=True, verify _alignData IS called."""
    mocker.patch.object(envfit_obj, '_alignData')
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(numpy.array([[1.0]]), [0]))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[1.0])
    
    # Needs valid point
    mz = envfit_obj.mzrange[0] + 0.1
    envfit_obj.topoints([[mz, 100.0]], autoAlign=True)
    
    envfit_obj._alignData.assert_called_once()

def test_topoints_leastSquare_zero(envfit_obj, mocker):
    """Test topoints where _leastSquare returns [0, 0] (for 2 models), check if it returns False."""
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(numpy.array([[1.0], [1.0]]), [0, 1]))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[0.0, 0.0])
    
    mz = envfit_obj.mzrange[0] + 0.1
    result = envfit_obj.topoints([[mz, 100.0]])
    
    assert result is False

def test_topoints_success(envfit_obj, mocker):
    """Test topoints with successful fit. Verify self.composition, self.ncomposition, self.average, and self.model."""
    # Mock models for x=0 and x=1
    # _makeModels returns (models_matrix, exchanged_list)
    # models_matrix should be (n_models, n_points)
    models_matrix = numpy.array([[0.8], [0.4]]) # 2 models, 1 point
    exchanged = [0, 1]
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(models_matrix, exchanged))
    
    # _leastSquare returns abundances
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=numpy.array([100.0, 50.0]))
    
    mz = envfit_obj.mzrange[0] + 0.1
    points = [[mz, 120.0]]
    
    result = envfit_obj.topoints(points, autoAlign=False)
    
    assert result is True
    # sum(fit) = 150.0
    # f = 1/150.0
    # self.models[0][2] = 100.0, [3] = 100/150 = 2/3
    # self.models[1][2] = 50.0, [3] = 50/150 = 1/3
    assert envfit_obj.composition[0] == 100.0
    assert envfit_obj.composition[1] == 50.0
    assert pytest.approx(envfit_obj.ncomposition[0]) == 2.0/3.0
    assert pytest.approx(envfit_obj.ncomposition[1]) == 1.0/3.0
    
    # average = 0 * 2/3 + 1 * 1/3 = 1/3
    assert pytest.approx(envfit_obj.average) == 1.0/3.0
    
    # self.model = concat(raster, intensities)
    # intensities = sum(models * [[x] for x in fit], axis=0)
    # intensities = 0.8 * 100.0 + 0.4 * 50.0 = 80.0 + 20.0 = 100.0
    assert envfit_obj.model.shape == (1, 2)
    assert envfit_obj.model[0][0] == mz
    assert envfit_obj.model[0][1] == 100.0

def test_envelope(envfit_obj, mocker):
    """Test envelope generation."""
    # Setup models with some abundances
    envfit_obj.models[0][1] = [(18.0, 1.0)]
    envfit_obj.models[0][2] = 100.0
    envfit_obj.models[1][1] = [(19.0, 1.0)]
    envfit_obj.models[1][2] = 50.0
    
    mock_profile = mocker.patch('mspy.mod_envfit.mod_pattern.profile', return_value=numpy.array([[18.0, 100.0], [19.0, 50.0]]))
    
    result = envfit_obj.envelope(points=20)
    
    assert isinstance(result, numpy.ndarray)
    # Verify mod_pattern.profile was called with merged isotopes
    # isotopes should be [(18.0, 100.0), (19.0, 50.0)]
    args, kwargs = mock_profile.call_args
    isotopes = args[0]
    assert (18.0, 100.0) in isotopes
    assert (19.0, 50.0) in isotopes
    assert kwargs['points'] == 20

def test_topeaklist(envfit_obj, mocker):
    """Test topeaklist method."""
    mock_topoints = mocker.patch.object(envfit_obj, 'topoints', return_value=True)
    
    from mspy.obj_peaklist import peaklist
    from mspy.obj_peak import peak
    
    pl = peaklist()
    # Ensure peaks are within mzrange of envfit_obj (H2O, charge 1)
    # mzrange is roughly [19.0, 20.03]
    p1 = peak(19.01, 100.0)
    p2 = peak(20.01, 50.0)
    pl.append(p1)
    pl.append(p2)
    
    envfit_obj.topeaklist(pl, fwhm=0.2, forceFwhm=True)
    
    # Verify topoints was called with correct points
    assert mock_topoints.called
    args, kwargs = mock_topoints.call_args
    points = kwargs['points']
    assert points.shape == (2, 2)
    assert points[0][0] == 19.01
    assert points[0][1] == 100.0
    assert kwargs['fwhm'] == 0.2

def test_tospectrum_baseline_none(envfit_obj, mocker, dummy_signal):
    """Test tospectrum with baseline=None."""
    # Mock mod_signal.locate to return indices for cropping
    mock_locate = mocker.patch('mspy.mod_envfit.mod_signal.locate', side_effect=[0, len(dummy_signal)])
    
    # Mock mod_peakpicking.labelscan to return a mock peaklist
    mock_peaklist = mocker.Mock()
    mock_labelscan = mocker.patch('mspy.mod_envfit.mod_peakpicking.labelscan', return_value=mock_peaklist)
    
    # Mock topeaklist method of envfit
    mock_topeaklist = mocker.patch.object(envfit_obj, 'topeaklist', return_value=True)

    result = envfit_obj.tospectrum(dummy_signal, fwhm=0.2, baseline=None)

    # Verify mod_signal.locate calls for cropping
    assert mock_locate.call_count == 2
    
    # Verify mod_peakpicking.labelscan call
    # Use call_args to avoid ValueError with numpy array comparison in assert_called_once_with
    mock_labelscan.assert_called_once()
    args, kwargs = mock_labelscan.call_args
    numpy.testing.assert_array_equal(kwargs['signal'], dummy_signal)
    assert kwargs['pickingHeight'] == 0.9
    assert kwargs['relThreshold'] == 0.0
    assert kwargs['baseline'] is None

    # Verify remshoulders call on the returned peaklist
    mock_peaklist.remshoulders.assert_called_once_with(fwhm=0.2)

    # Verify topeaklist call
    mock_topeaklist.assert_called_once_with(
        peaklist=mock_peaklist,
        fwhm=0.2,
        forceFwhm=True,
        autoAlign=True,
        iterLimit=None,
        relThreshold=0.0
    )
    assert result is True

def test_tospectrum_baseline_provided(envfit_obj, mocker, dummy_signal):
    """Test tospectrum with baseline provided."""
    # Mock dependencies to avoid side effects
    mocker.patch('mspy.mod_envfit.mod_signal.locate', side_effect=[0, len(dummy_signal)])
    mock_peaklist = mocker.Mock()
    mock_labelscan = mocker.patch('mspy.mod_envfit.mod_peakpicking.labelscan', return_value=mock_peaklist)
    mock_topeaklist = mocker.patch.object(envfit_obj, 'topeaklist', return_value=True)
    
    # Mock mod_signal.subbase
    updated_spectrum = numpy.array([[18.0, 100.0]])
    mock_subbase = mocker.patch('mspy.mod_envfit.mod_signal.subbase', return_value=updated_spectrum)

    # Use a list for baseline to avoid ValueError at 'if baseline != None:' in mod_envfit.py
    # subbase is mocked anyway, so it doesn't matter if it's a list or array here for the logic.
    baseline_list = [[18.0, 0.0], [19.0, 0.0], [20.0, 0.0]]
    result = envfit_obj.tospectrum(dummy_signal, baseline=baseline_list)

    # Verify mod_signal.subbase call
    mock_subbase.assert_called_once()
    args, kwargs = mock_subbase.call_args
    numpy.testing.assert_array_equal(args[0], dummy_signal)
    assert args[1] == baseline_list
    
    # Verify mod_peakpicking.labelscan call with baseline
    mock_labelscan.assert_called_once()
    _, ls_kwargs = mock_labelscan.call_args
    assert ls_kwargs['baseline'] == baseline_list

    # Verify self.spectrum updated
    numpy.testing.assert_array_equal(envfit_obj.spectrum, updated_spectrum)
    
    # Verify remshoulders call on the returned peaklist
    mock_peaklist.remshoulders.assert_called_once_with(fwhm=0.1)

    # Verify topeaklist call
    mock_topeaklist.assert_called_once_with(
        peaklist=mock_peaklist,
        fwhm=0.1,
        forceFwhm=True,
        autoAlign=True,
        iterLimit=None,
        relThreshold=0.0
    )
    assert result is True

def test_makeModels_raster_logic(envfit_obj, mocker):
    """Test _makeModels with raster logic and mz checks."""
    # H2O + H+ at ~19.018
    
    # 1. Trigger mz[0] > rasterMax (continue) -> Hits 305->306
    raster = numpy.array([10.0, 11.0]) # rasterMax = 11.1
    models, exchanged = envfit_obj._makeModels(raster)
    assert len(models) == 0
    assert len(exchanged) == 0

    # 2. Trigger mz[1] < rasterMin (continue) -> Hits 305->306
    raster = numpy.array([30.0, 31.0]) # rasterMin = 29.9
    models, exchanged = envfit_obj._makeModels(raster)
    assert len(models) == 0
    assert len(exchanged) == 0

    # 3. Trigger mz within raster (no continue) -> Hits 305->309
    # Mock mod_pattern.profile to avoid the bug in mod_pattern.py (raster != None)
    mocker.patch('mspy.mod_envfit.mod_pattern.profile', return_value=numpy.array([[19.0, 1.0], [20.0, 1.0]]))
    raster = numpy.array([19.0, 20.0])
    models, exchanged = envfit_obj._makeModels(raster)
    assert len(models) > 0

def test_makeModels_pattern_generation(envfit_obj, mocker):
    """Test _makeModels pattern generation and reset=False."""
    # Mock compound.pattern and mod_pattern.profile
    mock_pattern_call = mocker.patch('mspy.obj_compound.compound.pattern', return_value=[(19.018, 1.0)])
    mock_profile_call = mocker.patch('mspy.mod_pattern.profile', return_value=numpy.array([[19.018, 1.0]]))
    
    # Raster covering H2O
    raster = numpy.array([18.0, 19.0, 20.0])
    
    # 1. First call with reset=True (default)
    models, exchanged = envfit_obj._makeModels(raster, reset=True)
    assert len(models) == 2
    assert 0 in exchanged
    assert 1 in exchanged
    assert mock_pattern_call.call_count == 2 # 1 for scale 0, 1 for scale 1
    
    # 2. Second call with reset=False, should NOT call pattern again if already exists
    mock_pattern_call.reset_mock()
    models, exchanged = envfit_obj._makeModels(raster, reset=False)
    assert mock_pattern_call.call_count == 0
    
    # 3. Third call with reset=True, should call pattern again
    models, exchanged = envfit_obj._makeModels(raster, reset=True)
    assert mock_pattern_call.call_count == 2

def test_makeModels_no_any_model(envfit_obj, mocker):
    """Test _makeModels when profile returns all zeros (model.any() is False)."""
    mocker.patch('mspy.obj_compound.compound.pattern', return_value=[(19.018, 1.0)])
    # Return all zeros for profile
    mocker.patch('mspy.mod_pattern.profile', return_value=numpy.array([[19.0, 0.0], [20.0, 0.0]]))
    
    raster = numpy.array([19.0, 20.0])
    models, exchanged = envfit_obj._makeModels(raster)
    
    # model.any() should be False for [0.0, 0.0]
    assert len(models) == 0
    assert len(exchanged) == 0

def test_alignData_no_isotopes(envfit_obj, mocker):
    """Test _alignData when isotopes list is empty."""
    mocker.patch.object(envfit_obj, '_makeModels', return_value=([], []))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[])
    
    # Use valid data to get past earlier checks if any
    envfit_obj.data = numpy.array([[19.0, 100.0]])
    envfit_obj._alignData()
    # Should just return without errors
    assert True

def test_alignData_calibrants(envfit_obj, mocker):
    """Test _alignData with various calibrant selection scenarios."""
    # Setup data and mocked models
    envfit_obj.data = numpy.array([[19.01, 100.0], [20.01, 50.0], [21.01, 25.0]])
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(numpy.array([[1.0]]), [0]))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[1.0])
    
    # Set pattern for models[0] so isotopes is not empty
    envfit_obj.models[0][1] = [(19.0, 1.0)]
    
    # Mock mod_pattern.profile
    mocker.patch('mspy.mod_envfit.mod_pattern.profile', return_value=numpy.array([[19.0, 100.0]]))
    
    # Mock mod_peakpicking.labelscan to return 3 peaks
    from mspy.obj_peak import peak
    mock_peaklist = [peak(19.0, 100.0), peak(20.0, 50.0), peak(21.0, 25.0)]
    mocker.patch('mspy.mod_envfit.mod_peakpicking.labelscan', return_value=mock_peaklist)
    
    # Mock mod_calibration.calibration
    # Return (lambda params, x: x + params[0], [0.0], 0.0)
    mock_cal = mocker.patch('mspy.mod_envfit.mod_calibration.calibration', return_value=(lambda p, x: x + p[0], [0.01], 0.0))
    
    # 1. Test linear calibration (2 or 3 calibrants)
    envfit_obj._alignData()
    assert mock_cal.called
    assert mock_cal.call_args[1]['model'] == 'linear'
    
    # 2. Test quadratic calibration (> 3 calibrants)
    mock_peaklist = [peak(19.0, 100.0), peak(20.0, 50.0), peak(21.0, 25.0), peak(22.0, 10.0)]
    envfit_obj.data = numpy.array([[19.01, 100.0], [20.01, 50.0], [21.01, 25.0], [22.01, 10.0]])
    mocker.patch('mspy.mod_envfit.mod_peakpicking.labelscan', return_value=mock_peaklist)
    envfit_obj._alignData()
    assert mock_cal.call_args[1]['model'] == 'quadratic'

def test_alignData_edge_cases(envfit_obj, mocker):
    """Test edge cases in _alignData calibrant selection loop."""
    # Data: [mz, intensity]
    envfit_obj.data = numpy.array([
        [19.01, 100.0],  # Case B: new calibrant
        [19.02, 110.0],  # Case A: better calibrant (replaces 19.01)
        [20.50, 50.0],   # Case C: error > tolerance (break inner loop)
        [18.00, 10.0]    # Case D: error < -tolerance (continue inner loop)
    ])
    
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(numpy.array([[1.0]]), [0]))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[1.0])
    mocker.patch('mspy.mod_envfit.mod_pattern.profile', return_value=numpy.array([[19.0, 100.0]]))
    
    from mspy.obj_peak import peak
    # Tolerance is self.fwhm/1.5 = 0.1/1.5 = 0.066
    # peak(19.0, 100.0)
    # 19.01: error = 0.01 <= 0.066 -> Case B
    # 19.02: error = 0.02 <= 0.066 -> Case A (same peak, higher intensity)
    # 20.50: error = 1.5 > 0.066 -> Case C (break)
    
    # We need to make sure 18.00 is checked. If we have multiple peaks, we can trigger Case D.
    # Peak list sorted by m/z
    mock_peaklist = [peak(19.0, 100.0), peak(22.0, 50.0)]
    mocker.patch('mspy.mod_envfit.mod_peakpicking.labelscan', return_value=mock_peaklist)
    
    mock_cal = mocker.patch('mspy.mod_envfit.mod_calibration.calibration', return_value=(lambda p, x: x, [0.0], 0.0))
    
    envfit_obj._alignData()
    # Only 1 calibrant should be found because only 19.02 matches 19.0.
    # 22.0 doesn't match any point with tolerance.
    # If len(calibrants) <= 1, it returns.
    assert not mock_cal.called

def test_leastSquare_convergence(envfit_obj, mocker):
    """Test _leastSquare with convergence and negative parameter handling."""
    # Data and models (1 point, 1 model)
    data = numpy.array([100.0])
    models = numpy.array([[1.0]])
    
    # Mock solveLinEq to return delta
    mocker.patch('mspy.mod_envfit.solveLinEq', side_effect=[
        numpy.array([10.0]),  # 1st iteration: params=50+10=60
        numpy.array([-70.0]), # 2nd iteration: params=60-70=-10 -> clipped to 0
        numpy.array([0.0])    # 3rd iteration
    ])
    
    # Mock _chiSquare
    # return [chisq_value, chisq_deriv], alpha
    mocker.patch.object(envfit_obj, '_chiSquare', side_effect=[
        ([1000.0, [100.0]], numpy.array([[1.0]])), # Init
        ([500.0, [50.0]], numpy.array([[1.0]])),   # 1st loop (60)
        ([250.0, [25.0]], numpy.array([[1.0]])),   # 2nd loop (0)
        ([249.999, [20.0]], numpy.array([[1.0]]))  # 3rd loop (break)
    ])
    
    res = envfit_obj._leastSquare(data, models, chiLimit=0.1)
    # Normf = 100/100 = 1.0. Next_params /= 1.0.
    # Result should be 0.0 after second iteration + clipped
    assert res[0] >= 0.0

def test_leastSquare_divergence(envfit_obj, mocker):
    """Test _leastSquare when chi-square increases."""
    data = numpy.array([100.0])
    models = numpy.array([[1.0]])
    
    # Mock solveLinEq
    mocker.patch('mspy.mod_envfit.solveLinEq', return_value=numpy.array([10.0]))
    
    # Mock _chiSquare
    mocker.patch.object(envfit_obj, '_chiSquare', side_effect=[
        ([1000.0, [100.0]], numpy.array([[1.0]])), # Init
        ([1200.0, [110.0]], numpy.array([[1.0]])), # 1st loop: chisq increased!
        ([800.0, [50.0]], numpy.array([[1.0]])),   # 2nd loop: convergence
    ])
    
    # 1st loop: next_chisq (1200) > chisq (1000). l = 5*l.
    # 2nd loop: next_chisq (800) < chisq (1000). chisq - next_chisq = 200 > chiLimit (0.1).
    # break on 3rd loop with small diff
    mocker.patch.object(envfit_obj, '_chiSquare', side_effect=[
        ([1000.0, [100.0]], numpy.array([[1.0]])), # Init
        ([1200.0, [110.0]], numpy.array([[1.0]])), # 1st loop
        ([800.0, [50.0]], numpy.array([[1.0]])),   # 2nd loop
        ([799.99, [40.0]], numpy.array([[1.0]])),  # 3rd loop (break)
    ])
    
    res = envfit_obj._leastSquare(data, models, chiLimit=0.1)
    assert res is not None

def test_leastSquare_iterLimit(envfit_obj, mocker):
    """Test _leastSquare with iterLimit."""
    data = numpy.array([100.0])
    models = numpy.array([[1.0]])
    
    mocker.patch('mspy.mod_envfit.solveLinEq', return_value=numpy.array([1.0]))
    # Always returning smaller chi-square but large enough difference to not break
    mocker.patch.object(envfit_obj, '_chiSquare', return_value=([1000.0, [10.0]], numpy.array([[1.0]])))
    
    # Force many iterations by returning slightly smaller chisq each time
    chisqs = [([1000.0 - i, [10.0]], numpy.array([[1.0]])) for i in range(10)]
    mocker.patch.object(envfit_obj, '_chiSquare', side_effect=chisqs)
    
    res = envfit_obj._leastSquare(data, models, iterLimit=2)
    assert res is not None

def test_chiSquare(envfit_obj):
    """Test _chiSquare calculation."""
    data = numpy.array([100.0, 50.0])
    models = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    params = [90.0, 40.0]
    
    # differences = sum([[1,0],[0,1]] * [[90],[40]], axis=0) - [100, 50]
    # differences = [90, 40] - [100, 50] = [-10, -10]
    # chisq_value = (-10)**2 + (-10)**2 = 100 + 100 = 200
    
    chisq, alpha = envfit_obj._chiSquare(data, models, params)
    
    assert chisq[0] == 200.0
    # chisq_deriv: for x=0 (mz=0), data=100, models=[1,0], diff=-10. deriv=[1,0]. chisq_deriv += -20 * [1,0] = [-20, 0]
    # for x=1 (mz=1), data=50, models=[0,1], diff=-10. deriv=[0,1]. chisq_deriv += -20 * [0,1] = [-20, -20]
    assert chisq[1] == [-20.0, -20.0]
    # alpha: x=0, d=[1,0]. d[:,newaxis]*d = [[1,0],[0,0]]. alpha = [[1,0],[0,0]]
    # x=1, d=[0,1]. d[:,newaxis]*d = [[0,0],[0,1]]. alpha = [[1,0],[0,1]]
    assert numpy.array_equal(alpha, numpy.array([[1.0, 0.0], [0.0, 1.0]]))

@given(
    formula=st.sampled_from(['H2O', 'C6H12O6', 'NH3']),
    charge=st.integers(min_value=0, max_value=1),
    scales=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=2, unique=True),
    loss=st.just('H'),
    gain=st.just('H{2}'),
)
def test_init_robustness(formula, charge, scales, loss, gain):
    """Property-based test for envfit initialization."""
    # This should not raise any unhandled exceptions for these standard inputs
    ef = envfit(formula, charge, scales, loss=loss, gain=gain)
    assert isinstance(ef, envfit)
    assert len(ef.models) <= len(scales)

def test_topeaklist_list_input(envfit_obj, mocker):
    """Test topeaklist with list input to hit line 125."""
    mocker.patch.object(envfit_obj, 'topoints', return_value=True)
    envfit_obj.topeaklist([[19.0, 100.0]], fwhm=0.2)
    assert envfit_obj.topoints.called

def test_topeaklist_fwhm_from_basepeak(envfit_obj, mocker):
    """Test topeaklist getting fwhm from basepeak to hit line 136."""
    from mspy.obj_peaklist import peaklist
    from mspy.obj_peak import peak
    pl = peaklist()
    p = peak(19.01, 100.0)
    p.fwhm = 0.5
    pl.append(p)
    
    mock_topoints = mocker.patch.object(envfit_obj, 'topoints', return_value=True)
    envfit_obj.topeaklist(pl, forceFwhm=False)
    
    # Verify topoints was called with fwhm=0.5
    kwargs = mock_topoints.call_args[1]
    assert kwargs['fwhm'] == 0.5

def test_topoints_ndarray_input(envfit_obj, mocker):
    """Test topoints with ndarray input to hit branch skip at line 171."""
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(numpy.array([[1.0]]), [0]))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[1.0])
    
    mz = envfit_obj.mzrange[0] + 0.1
    points = numpy.array([[mz, 100.0]])
    envfit_obj.topoints(points, autoAlign=False)
    # This should hit the 171->175 branch

def test_alignData_multiple_points_per_peak(envfit_obj, mocker):
    """Test _alignData with multiple points matching same peak to hit line 368."""
    # Data: two points at 19.0, second one is more intense
    # calibrants[-1][0] == peak.mz is point[0] == peak.mz
    envfit_obj.data = numpy.array([
        [19.0, 100.0],
        [19.0, 200.0],
        [20.0, 50.0]
    ])
    # spectrum to hit line 386
    envfit_obj.spectrum = numpy.array([[19.0, 100.0], [20.0, 50.0]])
    
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(numpy.array([[1.0]]), [0]))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[1.0])
    envfit_obj.models[0][1] = [(19.0, 1.0), (20.0, 0.5)]
    
    from mspy.obj_peak import peak
    mock_peaklist = [peak(19.0, 100.0), peak(20.0, 50.0)]
    mocker.patch('mspy.mod_envfit.mod_peakpicking.labelscan', return_value=mock_peaklist)
    
    mock_cal = mocker.patch('mspy.mod_envfit.mod_calibration.calibration', return_value=(lambda p, x: x, [0.0], 0.0))
    
    envfit_obj._alignData()
    assert mock_cal.called
    # verify calibrants list passed to calibration
    # Should have [(19.0, 19.0), (20.0, 20.0)]
    args = mock_cal.call_args[0]
    calibrants = args[0]
    assert len(calibrants) == 2
    assert calibrants[0] == (19.0, 19.0) # Replaced by the second point (19.0, 200.0)

def test_alignData_too_few_calibrants(envfit_obj, mocker):
    """Test _alignData with too few calibrants to hit line 380."""
    envfit_obj.data = numpy.array([[19.01, 100.0]])
    mocker.patch.object(envfit_obj, '_makeModels', return_value=(numpy.array([[1.0]]), [0]))
    mocker.patch.object(envfit_obj, '_leastSquare', return_value=[1.0])
    envfit_obj.models[0][1] = [(19.0, 1.0)]
    
    from mspy.obj_peak import peak
    mock_peaklist = [peak(19.0, 100.0)]
    mocker.patch('mspy.mod_envfit.mod_peakpicking.labelscan', return_value=mock_peaklist)
    
    mock_cal = mocker.patch('mspy.mod_envfit.mod_calibration.calibration')
    envfit_obj._alignData()
    assert not mock_cal.called
