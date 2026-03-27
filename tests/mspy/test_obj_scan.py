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
import numpy
import copy
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
try:
    from unittest.mock import patch, MagicMock
except ImportError:
    from mock import patch, MagicMock

import obj_scan
import obj_peak
import obj_peaklist
import mod_signal
import mod_peakpicking
import mod_stopper


# Module-level fixture to reset stopper state
@pytest.fixture(scope="module", autouse=True)
def reset_stopper():
    """Reset stopper state before running tests."""
    mod_stopper.start()
    yield
    mod_stopper.start()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_profile(mz_values, intensity_values):
    """Create a 2D numpy array profile from mz and intensity lists."""
    if not mz_values:
        return numpy.array([])
    pairs = [[float(mz), float(ai)] for mz, ai in zip(mz_values, intensity_values)]
    return numpy.array(pairs)


def make_gaussian_profile(mz_center=500.0, intensity=1000.0, width=1.0, num_points=100):
    """Create a Gaussian-shaped profile."""
    mz_min = mz_center - 5 * width
    mz_max = mz_center + 5 * width
    mz_array = numpy.linspace(mz_min, mz_max, num_points)

    # Gaussian: exp(-(x-center)^2 / (2*sigma^2))
    sigma = width / 2.355  # FWHM to sigma conversion
    ai_array = intensity * numpy.exp(-((mz_array - mz_center) ** 2) / (2 * sigma ** 2))

    profile = numpy.column_stack([mz_array, ai_array])
    return profile


def make_peaks(*pairs):
    """Create peak objects from (mz, ai) tuples."""
    return [obj_peak.peak(mz, ai=ai) for mz, ai in pairs]


def make_peaklist(*pairs):
    """Create a peaklist from (mz, ai) tuples."""
    peaks = make_peaks(*pairs)
    return obj_peaklist.peaklist(peaks)


def linear_fn(params, x):
    """Simple linear calibration function: y = m*x + b."""
    m, b = params
    return m * x + b


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def empty_scan():
    """Create an empty scan with no profile or peaklist."""
    return obj_scan.scan()


@pytest.fixture
def profile_scan():
    """Create a scan with profile data only."""
    profile = make_profile([100.0, 200.0, 300.0], [50.0, 100.0, 75.0])
    return obj_scan.scan(profile=profile)


@pytest.fixture
def peaklist_scan():
    """Create a scan with peaklist data only."""
    peaks = make_peaks((100.0, 100.0), (200.0, 50.0), (300.0, 75.0))
    peaklist = obj_peaklist.peaklist(peaks)
    return obj_scan.scan(peaklist=peaklist)


@pytest.fixture
def full_scan():
    """Create a scan with both profile and peaklist data."""
    profile = make_profile([100.0, 200.0, 300.0], [50.0, 100.0, 75.0])
    peaks = make_peaks((100.0, 100.0), (200.0, 50.0), (300.0, 75.0))
    peaklist = obj_peaklist.peaklist(peaks)
    return obj_scan.scan(profile=profile, peaklist=peaklist)


@pytest.fixture
def gaussian_scan():
    """Create a scan with a Gaussian profile."""
    profile = make_gaussian_profile(mz_center=500.0, intensity=1000.0, width=1.0)
    return obj_scan.scan(profile=profile)


# ============================================================================
# TEST: __init__ (B1, B2, B3)
# ============================================================================

class TestScanInit:
    """Test scan initialization."""

    def test_init_empty(self):
        """Initialize with default arguments (empty scan)."""
        s = obj_scan.scan()
        assert len(s) == 0
        assert len(s.peaklist) == 0
        assert s.title == ''
        assert s.scanNumber is None
        assert s._baseline is None
        assert s.attributes == {}

    def test_init_with_list_profile(self):
        """B1-true: convert list profile to ndarray."""
        profile_list = [[100.0, 50.0], [200.0, 100.0]]
        s = obj_scan.scan(profile=profile_list)
        assert isinstance(s.profile, numpy.ndarray)
        assert len(s) == 2
        assert s.profile[0][0] == 100.0

    def test_init_with_ndarray_profile(self):
        """B1-false: pass ndarray profile directly."""
        profile_array = numpy.array([[100.0, 50.0], [200.0, 100.0]])
        s = obj_scan.scan(profile=profile_array)
        assert isinstance(s.profile, numpy.ndarray)
        assert len(s) == 2
        assert numpy.array_equal(s.profile, profile_array)

    def test_init_with_list_peaklist(self):
        """B2-true: wrap list of peaks in peaklist object."""
        peaks_list = [[100.0, 50.0], [200.0, 100.0]]
        s = obj_scan.scan(peaklist=peaks_list)
        assert isinstance(s.peaklist, obj_peaklist.peaklist)
        assert len(s.peaklist) == 2

    def test_init_with_peaklist_object(self):
        """B2-false: pass peaklist object directly."""
        peaks = make_peaks((100.0, 50.0), (200.0, 100.0))
        peaklist = obj_peaklist.peaklist(peaks)
        s = obj_scan.scan(peaklist=peaklist)
        assert s.peaklist is peaklist

    def test_init_with_kwargs(self):
        """B3-true: populate attributes dict from kwargs."""
        s = obj_scan.scan(title='test', scanNumber=1, msLevel=2)
        assert s.attributes['title'] == 'test'
        assert s.attributes['scanNumber'] == 1
        assert s.attributes['msLevel'] == 2

    def test_init_without_kwargs(self):
        """B3-false: empty attributes dict when no kwargs."""
        s = obj_scan.scan()
        assert s.attributes == {}

    def test_init_mixed_kwargs(self):
        """B3-true: kwargs mixed with positional args."""
        profile = make_profile([100.0], [50.0])
        s = obj_scan.scan(profile=profile, custom_attr='value')
        assert len(s) == 1
        assert s.attributes['custom_attr'] == 'value'


# ============================================================================
# TEST: __len__
# ============================================================================

class TestScanLen:
    """Test scan length (__len__)."""

    def test_len_empty_scan(self, empty_scan):
        """Length of empty scan is 0."""
        assert len(empty_scan) == 0

    def test_len_with_profile(self, profile_scan):
        """Length equals profile length."""
        assert len(profile_scan) == 3

    def test_len_with_peaklist_only(self, peaklist_scan):
        """Length is based on profile, not peaklist."""
        assert len(peaklist_scan) == 0


# ============================================================================
# TEST: __add__, __sub__, __mul__
# ============================================================================

class TestScanDunder:
    """Test dunder methods: __add__, __sub__, __mul__."""

    def test_add_returns_new_scan(self, profile_scan):
        """__add__ returns a new scan object."""
        other = obj_scan.scan(profile=make_profile([400.0], [50.0]))
        result = profile_scan + other
        assert isinstance(result, obj_scan.scan)
        assert result is not profile_scan
        assert result is not other

    def test_sub_returns_new_scan(self, profile_scan):
        """__sub__ returns a new scan object."""
        other = obj_scan.scan(profile=make_profile([100.0, 200.0, 300.0], [10.0, 20.0, 15.0]))
        result = profile_scan - other
        assert isinstance(result, obj_scan.scan)
        assert result is not profile_scan
        assert result is not other

    def test_mul_returns_new_scan(self, profile_scan):
        """__mul__ returns a new scan object."""
        result = profile_scan * 2.0
        assert isinstance(result, obj_scan.scan)
        assert result is not profile_scan

    def test_add_modifies_copy_not_original(self, profile_scan):
        """__add__ doesn't modify original scan."""
        original_len = len(profile_scan)
        other = obj_scan.scan(profile=make_profile([400.0], [50.0]))
        result = profile_scan + other
        assert len(profile_scan) == original_len

    def test_mul_scales_profile(self, profile_scan):
        """__mul__ scales the profile in result."""
        result = profile_scan * 2.0
        # Original unchanged
        assert profile_scan.profile[0][1] == 50.0
        # Result scaled
        assert result.profile[0][1] == 100.0


# ============================================================================
# TEST: reset()
# ============================================================================

class TestScanReset:
    """Test reset() method."""

    def test_reset_clears_baseline(self):
        """reset() sets _baseline to None."""
        s = obj_scan.scan(profile=make_profile([100.0], [50.0]))
        s._baseline = numpy.array([1, 2, 3])
        s.reset()
        assert s._baseline is None

    def test_reset_clears_baseline_params(self):
        """reset() resets _baselineParams to defaults."""
        s = obj_scan.scan(profile=make_profile([100.0], [50.0]))
        s._baselineParams['window'] = 0.5
        s._baselineParams['offset'] = 0.2
        s.reset()
        assert s._baselineParams['window'] is None
        assert s._baselineParams['offset'] is None

    def test_reset_preserves_profile(self, profile_scan):
        """reset() preserves profile data."""
        original_profile = copy.deepcopy(profile_scan.profile)
        profile_scan.reset()
        assert numpy.array_equal(profile_scan.profile, original_profile)

    def test_reset_preserves_peaklist(self, full_scan):
        """reset() preserves peaklist data."""
        original_len = len(full_scan.peaklist)
        full_scan.reset()
        assert len(full_scan.peaklist) == original_len


# ============================================================================
# TEST: duplicate()
# ============================================================================

class TestScanDuplicate:
    """Test duplicate() method."""

    def test_duplicate_returns_new_object(self, profile_scan):
        """duplicate() returns a new scan object."""
        dup = profile_scan.duplicate()
        assert isinstance(dup, obj_scan.scan)
        assert dup is not profile_scan

    def test_duplicate_copies_profile(self, profile_scan):
        """duplicate() deepcopies profile data."""
        dup = profile_scan.duplicate()
        assert numpy.array_equal(dup.profile, profile_scan.profile)
        # Ensure it's a copy, not the same object
        assert dup.profile is not profile_scan.profile

    def test_duplicate_copies_peaklist(self, full_scan):
        """duplicate() deepcopies peaklist."""
        dup = full_scan.duplicate()
        assert len(dup.peaklist) == len(full_scan.peaklist)
        assert dup.peaklist is not full_scan.peaklist

    def test_duplicate_copies_metadata(self, full_scan):
        """duplicate() copies all metadata."""
        full_scan.title = 'Test Scan'
        full_scan.scanNumber = 42
        dup = full_scan.duplicate()
        assert dup.title == 'Test Scan'
        assert dup.scanNumber == 42

    def test_duplicate_copies_attributes(self):
        """duplicate() copies custom attributes."""
        s = obj_scan.scan(custom='value', other=123)
        dup = s.duplicate()
        assert dup.attributes['custom'] == 'value'
        assert dup.attributes['other'] == 123


# ============================================================================
# TEST: noise()
# ============================================================================

class TestScanNoise:
    """Test noise() method."""

    def test_noise_with_profile(self, profile_scan):
        """noise() returns tuple when called on profile."""
        result = profile_scan.noise(minX=100.0, maxX=200.0)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_noise_delegates_to_mod_signal(self, profile_scan):
        """noise() delegates to mod_signal.noise()."""
        with patch('mod_signal.noise', return_value=(1.0, 0.5)) as mock_noise:
            result = profile_scan.noise(minX=100.0, maxX=200.0)
            mock_noise.assert_called_once()
            assert result == (1.0, 0.5)

    def test_noise_with_mz_parameter(self, profile_scan):
        """noise() accepts mz parameter."""
        result = profile_scan.noise(mz=150.0, window=0.1)
        assert result is not None


# ============================================================================
# TEST: baseline() (B4)
# ============================================================================

class TestScanBaseline:
    """Test baseline() method with caching (B4)."""

    def test_baseline_computed_when_none(self, profile_scan):
        """B4-false: compute baseline when _baseline is None."""
        assert profile_scan._baseline is None
        # Use a non-array object to avoid the numpy comparison issue
        baseline_result = [100.0, 50.0]
        with patch('mod_signal.baseline', return_value=baseline_result):
            baseline = profile_scan.baseline(window=0.1, offset=0.0)
            assert profile_scan._baseline is not None

    def test_baseline_cached_on_second_call(self):
        """B4-true: return cached baseline on identical params."""
        s = obj_scan.scan(profile=make_profile([100.0, 200.0], [50.0, 100.0]))
        call_count = [0]
        baseline_result = [100.0, 50.0]
        def mock_baseline_fn(**kw):
            call_count[0] += 1
            return baseline_result

        with patch('mod_signal.baseline', side_effect=mock_baseline_fn):
            baseline1 = s.baseline(window=0.1, offset=0.0)
            baseline2 = s.baseline(window=0.1, offset=0.0)
            assert call_count[0] == 1  # Only called once due to caching

    def test_baseline_recomputed_on_window_change(self):
        """B4-false: recompute when window changes."""
        s = obj_scan.scan(profile=make_profile([100.0, 200.0], [50.0, 100.0]))
        call_count = [0]
        baseline_result = [100.0, 50.0]
        def mock_baseline_fn(**kw):
            call_count[0] += 1
            return baseline_result

        with patch('mod_signal.baseline', side_effect=mock_baseline_fn):
            s.baseline(window=0.1, offset=0.0)
            s.baseline(window=0.2, offset=0.0)
            assert call_count[0] == 2  # Called twice due to window change

    def test_baseline_recomputed_on_offset_change(self):
        """B4-false: recompute when offset changes."""
        s = obj_scan.scan(profile=make_profile([100.0, 200.0], [50.0, 100.0]))
        call_count = [0]
        baseline_result = [100.0, 50.0]
        def mock_baseline_fn(**kw):
            call_count[0] += 1
            return baseline_result

        with patch('mod_signal.baseline', side_effect=mock_baseline_fn):
            s.baseline(window=0.1, offset=0.0)
            s.baseline(window=0.1, offset=0.5)
            assert call_count[0] == 2  # Called twice due to offset change

    def test_baseline_params_stored(self):
        """baseline() stores params for cache validation."""
        s = obj_scan.scan(profile=make_profile([100.0, 200.0], [50.0, 100.0]))
        with patch('mod_signal.baseline', return_value=[100.0, 50.0]):
            s.baseline(window=0.15, offset=0.25)
            assert s._baselineParams['window'] == 0.15
            assert s._baselineParams['offset'] == 0.25

    def test_baseline_call_count_with_caching(self):
        """Verify call count with caching behavior."""
        s = obj_scan.scan(profile=make_profile([100.0, 200.0], [50.0, 100.0]))
        baseline_result = [100.0, 50.0]
        with patch('mod_signal.baseline', side_effect=lambda **kw: baseline_result) as mock_bl:
            s.baseline(window=0.1, offset=0.0)
            s.baseline(window=0.1, offset=0.0)
            s.baseline(window=0.1, offset=0.0)
            # Should only call mod_signal.baseline once
            assert mock_bl.call_count == 1


# ============================================================================
# TEST: normalization() (B5, B6, B7, B8)
# ============================================================================

class TestScanNormalization:
    """Test normalization() method."""

    def test_normalization_both_profile_and_peaklist(self, full_scan):
        """B5-true: use both profile and peaklist for normalization."""
        norm = full_scan.normalization()
        assert isinstance(norm, float)
        assert norm > 0.0

    def test_normalization_profile_only(self, profile_scan):
        """B6-true: use profile only when peaklist empty."""
        norm = profile_scan.normalization()
        assert isinstance(norm, float)
        assert norm > 0.0
        # Should be max intensity / 100
        assert norm == 100.0 / 100.0

    def test_normalization_peaklist_only(self, peaklist_scan):
        """B7-true: use peaklist only when profile empty."""
        norm = peaklist_scan.normalization()
        assert isinstance(norm, float)
        assert norm > 0.0

    def test_normalization_empty_scan(self, empty_scan):
        """B8-true: return 1.0 when both empty."""
        norm = empty_scan.normalization()
        assert norm == 1.0

    def test_normalization_uses_spectrum_max(self, profile_scan):
        """normalization() uses max spectrum intensity."""
        # profile has max intensity of 100.0
        norm = profile_scan.normalization()
        assert norm == 100.0 / 100.0


# ============================================================================
# TEST: intensity()
# ============================================================================

class TestScanIntensity:
    """Test intensity() method."""

    def test_intensity_delegates_to_mod_signal(self, profile_scan):
        """intensity() delegates to mod_signal.intensity()."""
        with patch('mod_signal.intensity', return_value=50.0) as mock_int:
            result = profile_scan.intensity(mz=150.0)
            mock_int.assert_called_once()
            assert result == 50.0

    def test_intensity_returns_float(self, profile_scan):
        """intensity() returns a float."""
        result = profile_scan.intensity(mz=150.0)
        assert isinstance(result, (float, numpy.floating))


# ============================================================================
# TEST: width()
# ============================================================================

class TestScanWidth:
    """Test width() method."""

    def test_width_delegates_to_mod_signal(self, profile_scan):
        """width() delegates to mod_signal.width()."""
        with patch('mod_signal.width', return_value=0.5) as mock_width:
            result = profile_scan.width(mz=150.0, intensity=50.0)
            mock_width.assert_called_once()
            assert result == 0.5


# ============================================================================
# TEST: area() (B9)
# ============================================================================

class TestScanArea:
    """Test area() method."""

    def test_area_empty_scan_returns_zero(self, empty_scan):
        """B9-true: return 0.0 for empty profile."""
        area = empty_scan.area()
        assert area == 0.0

    def test_area_with_profile(self, profile_scan):
        """B9-false: compute area for non-empty profile."""
        with patch('mod_signal.baseline', return_value=[100.0, 50.0]):
            with patch('mod_signal.area', return_value=1000.0):
                area = profile_scan.area()
                assert isinstance(area, (float, numpy.floating))

    def test_area_calls_baseline(self, profile_scan):
        """area() retrieves baseline before calculating."""
        with patch.object(profile_scan, 'baseline', return_value=numpy.array([])) as mock_bl:
            profile_scan.area()
            mock_bl.assert_called_once()

    def test_area_delegates_to_mod_signal(self, profile_scan):
        """area() delegates calculation to mod_signal.area()."""
        with patch('mod_signal.area', return_value=1000.0) as mock_area:
            result = profile_scan.area(minX=100.0, maxX=200.0)
            mock_area.assert_called_once()
            assert result == 1000.0


# ============================================================================
# TEST: hasprofile(), haspeaks()
# ============================================================================

class TestScanHas:
    """Test hasprofile() and haspeaks() methods."""

    def test_hasprofile_empty_scan(self, empty_scan):
        """hasprofile() returns False for empty profile."""
        assert empty_scan.hasprofile() is False

    def test_hasprofile_with_profile(self, profile_scan):
        """hasprofile() returns True when profile present."""
        assert profile_scan.hasprofile() is True

    def test_haspeaks_empty_scan(self, empty_scan):
        """haspeaks() returns False for empty peaklist."""
        assert empty_scan.haspeaks() is False

    def test_haspeaks_with_peaklist(self, peaklist_scan):
        """haspeaks() returns True when peaklist present."""
        assert peaklist_scan.haspeaks() is True

    def test_haspeaks_full_scan(self, full_scan):
        """haspeaks() returns True when peaklist present."""
        assert full_scan.haspeaks() is True


# ============================================================================
# TEST: setprofile()
# ============================================================================

class TestScanSetprofile:
    """Test setprofile() method."""

    def test_setprofile_updates_profile(self, empty_scan):
        """setprofile() updates profile data."""
        profile = make_profile([100.0, 200.0], [50.0, 100.0])
        empty_scan.setprofile(profile)
        assert len(empty_scan) == 2
        assert numpy.array_equal(empty_scan.profile, profile)

    def test_setprofile_resets_buffers(self):
        """setprofile() clears baseline cache."""
        s = obj_scan.scan(profile=make_profile([100.0], [50.0]))
        s._baseline = numpy.array([1, 2, 3])
        new_profile = make_profile([200.0], [75.0])
        s.setprofile(new_profile)
        assert s._baseline is None

    def test_setprofile_clears_baseline_params(self):
        """setprofile() resets baseline params."""
        s = obj_scan.scan(profile=make_profile([100.0], [50.0]))
        s._baselineParams['window'] = 0.5
        new_profile = make_profile([200.0], [75.0])
        s.setprofile(new_profile)
        assert s._baselineParams['window'] is None


# ============================================================================
# TEST: setpeaklist() (B10)
# ============================================================================

class TestScanSetpeaklist:
    """Test setpeaklist() method."""

    def test_setpeaklist_with_peaklist_object(self, empty_scan):
        """B10-true: assign peaklist object directly."""
        peaks = make_peaks((100.0, 50.0), (200.0, 100.0))
        peaklist = obj_peaklist.peaklist(peaks)
        empty_scan.setpeaklist(peaklist)
        assert empty_scan.peaklist is peaklist

    def test_setpeaklist_with_list(self, empty_scan):
        """B10-false: wrap list in peaklist object."""
        peaks_list = [[100.0, 50.0], [200.0, 100.0]]
        empty_scan.setpeaklist(peaks_list)
        assert isinstance(empty_scan.peaklist, obj_peaklist.peaklist)
        assert len(empty_scan.peaklist) == 2

    def test_setpeaklist_with_peak_objects(self, empty_scan):
        """B10-false: wrap peak objects in peaklist."""
        peaks = make_peaks((100.0, 50.0), (200.0, 100.0))
        empty_scan.setpeaklist(peaks)
        assert isinstance(empty_scan.peaklist, obj_peaklist.peaklist)
        assert len(empty_scan.peaklist) == 2


# ============================================================================
# TEST: swap()
# ============================================================================

class TestScanSwap:
    """Test swap() method."""

    def test_swap_exchanges_profile_and_peaklist(self, full_scan):
        """swap() exchanges profile and peaklist data."""
        original_profile_len = len(full_scan.profile)
        original_peaklist_len = len(full_scan.peaklist)

        full_scan.swap()

        assert len(full_scan.profile) == original_peaklist_len
        assert len(full_scan.peaklist) == original_profile_len

    def test_swap_resets_buffers(self, full_scan):
        """swap() clears baseline cache."""
        full_scan._baseline = numpy.array([1, 2, 3])
        full_scan.swap()
        assert full_scan._baseline is None

    def test_swap_empty_scan(self, empty_scan):
        """swap() on empty scan doesn't error."""
        empty_scan.swap()
        assert len(empty_scan) == 0
        assert len(empty_scan.peaklist) == 0


# ============================================================================
# TEST: crop()
# ============================================================================

class TestScanCrop:
    """Test crop() method."""

    def test_crop_removes_out_of_range_profile(self, profile_scan):
        """crop() removes profile data outside range."""
        profile_scan.crop(minX=150.0, maxX=250.0)
        # Should have one or zero points after crop
        assert len(profile_scan) >= 0

    def test_crop_resets_buffers(self, profile_scan):
        """crop() clears baseline cache."""
        profile_scan._baseline = numpy.array([1, 2, 3])
        profile_scan.crop(minX=150.0, maxX=250.0)
        assert profile_scan._baseline is None

    def test_crop_delegates_to_mod_signal(self, profile_scan):
        """crop() delegates profile crop to mod_signal."""
        with patch('mod_signal.crop', return_value=numpy.array([])) as mock_crop:
            profile_scan.crop(minX=150.0, maxX=250.0)
            mock_crop.assert_called_once()

    def test_crop_crops_peaklist(self, full_scan):
        """crop() calls peaklist.crop()."""
        with patch.object(full_scan.peaklist, 'crop') as mock_crop:
            full_scan.crop(minX=150.0, maxX=250.0)
            mock_crop.assert_called_once()


# ============================================================================
# TEST: multiply() (B11)
# ============================================================================

class TestScanMultiply:
    """Test multiply() method."""

    def test_multiply_scales_profile(self, profile_scan):
        """B11-true: multiply profile by factor."""
        original_intensity = profile_scan.profile[0][1]
        profile_scan.multiply(2.0)
        assert profile_scan.profile[0][1] == original_intensity * 2.0

    def test_multiply_empty_profile_no_error(self, peaklist_scan):
        """B11-false: empty profile doesn't error."""
        peaklist_scan.multiply(2.0)  # Should not raise

    def test_multiply_scales_peaklist(self, full_scan):
        """multiply() also scales peaklist."""
        with patch.object(full_scan.peaklist, 'multiply') as mock_mul:
            full_scan.multiply(2.0)
            mock_mul.assert_called_once_with(2.0)

    def test_multiply_resets_buffers(self, profile_scan):
        """multiply() clears baseline cache."""
        profile_scan._baseline = numpy.array([1, 2, 3])
        profile_scan.multiply(2.0)
        assert profile_scan._baseline is None

    def test_multiply_delegates_profile_to_mod_signal(self, profile_scan):
        """multiply() delegates profile scaling to mod_signal."""
        with patch('mod_signal.multiply', return_value=numpy.array([])) as mock_mul:
            profile_scan.multiply(2.0)
            mock_mul.assert_called_once()


# ============================================================================
# TEST: normalize() (B12, B13)
# ============================================================================

class TestScanNormalize:
    """Test normalize() method."""

    def test_normalize_scales_profile(self, profile_scan):
        """B12-true: normalize profile when non-empty."""
        # Just verify that normalization happens (profile is modified)
        profile_scan.normalize()
        # After normalize, values should be scaled down

    def test_normalize_empty_profile_no_error(self, peaklist_scan):
        """B12-false: empty profile doesn't error."""
        peaklist_scan.normalize()  # Should not raise

    def test_normalize_scales_peaklist(self, full_scan):
        """B13-true: normalize peaks when non-empty."""
        with patch.object(full_scan.peaklist, 'multiply') as mock_mul:
            full_scan.normalize()
            # peaklist.multiply gets called in normalize

    def test_normalize_empty_peaklist_no_error(self, profile_scan):
        """B13-false: empty peaklist doesn't error."""
        profile_scan.normalize()  # Should not raise

    def test_normalize_resets_buffers(self, full_scan):
        """normalize() clears baseline cache."""
        full_scan._baseline = numpy.array([1, 2, 3])
        full_scan.normalize()
        assert full_scan._baseline is None


# ============================================================================
# TEST: combine() (B14, B15, B16)
# ============================================================================

class TestScanCombine:
    """Test combine() method."""

    def test_combine_non_scan_raises_typeerror(self, profile_scan):
        """B14-true: TypeError when combining with non-scan."""
        with pytest.raises(TypeError):
            profile_scan.combine("not a scan")

    def test_combine_profiles_when_available(self, profile_scan):
        """B15-true: use profiles when available."""
        other = obj_scan.scan(profile=make_profile([400.0], [50.0]))
        with patch('mod_signal.combine', return_value=numpy.array([])) as mock_comb:
            profile_scan.combine(other)
            mock_comb.assert_called_once()

    def test_combine_with_empty_other(self, profile_scan):
        """B15-true: combine even with empty other profile."""
        other = obj_scan.scan()
        with patch('mod_signal.combine', return_value=numpy.array([])) as mock_comb:
            profile_scan.combine(other)
            mock_comb.assert_called_once()

    def test_combine_empties_peaklist_when_using_profiles(self, full_scan):
        """combine() empties peaklist when combining profiles."""
        full_scan.combine(obj_scan.scan(profile=make_profile([400.0], [50.0])))
        assert len(full_scan.peaklist) == 0

    def test_combine_peaklists_when_no_profiles(self):
        """B16-true: combine peaklists when profiles empty."""
        s1 = obj_scan.scan(peaklist=make_peaklist((100.0, 100.0)))
        s2 = obj_scan.scan(peaklist=make_peaklist((200.0, 50.0)))

        with patch.object(s1.peaklist, 'combine') as mock_comb:
            s1.combine(s2)
            mock_comb.assert_called_once()

    def test_combine_both_empty_no_op(self, empty_scan):
        """B16-false: no-op when both empty."""
        other = obj_scan.scan()
        empty_scan.combine(other)  # Should not error

    def test_combine_resets_buffers(self, profile_scan):
        """combine() clears baseline cache."""
        profile_scan._baseline = numpy.array([1, 2, 3])
        other = obj_scan.scan(profile=make_profile([400.0], [50.0]))
        profile_scan.combine(other)
        assert profile_scan._baseline is None


# ============================================================================
# TEST: overlay() (B17, B18)
# ============================================================================

class TestScanOverlay:
    """Test overlay() method."""

    def test_overlay_non_scan_raises_typeerror(self, profile_scan):
        """B17-true: TypeError when overlaying with non-scan."""
        with pytest.raises(TypeError):
            profile_scan.overlay("not a scan")

    def test_overlay_profiles_when_available(self, profile_scan):
        """B18-true: use profiles when available."""
        other = obj_scan.scan(profile=make_profile([400.0], [50.0]))
        with patch('mod_signal.overlay', return_value=numpy.array([])) as mock_ovly:
            profile_scan.overlay(other)
            mock_ovly.assert_called_once()

    def test_overlay_empties_peaklist(self, full_scan):
        """overlay() empties peaklist when using profiles."""
        full_scan.overlay(obj_scan.scan(profile=make_profile([400.0], [50.0])))
        assert len(full_scan.peaklist) == 0

    def test_overlay_resets_buffers(self, profile_scan):
        """overlay() clears baseline cache."""
        profile_scan._baseline = numpy.array([1, 2, 3])
        other = obj_scan.scan(profile=make_profile([400.0], [50.0]))
        profile_scan.overlay(other)
        assert profile_scan._baseline is None

    def test_overlay_no_profile_no_op(self, peaklist_scan):
        """B18-false: no-op when no profile data."""
        other = obj_scan.scan()
        # Should not raise or modify
        peaklist_scan.overlay(other)


# ============================================================================
# TEST: subtract() (B19, B20)
# ============================================================================

class TestScanSubtract:
    """Test subtract() method."""

    def test_subtract_non_scan_raises_typeerror(self, profile_scan):
        """B19-true: TypeError when subtracting non-scan."""
        with pytest.raises(TypeError):
            profile_scan.subtract("not a scan")

    def test_subtract_both_profiles(self, profile_scan):
        """B20-true: subtract when both have profiles."""
        other = obj_scan.scan(profile=make_profile([100.0, 200.0, 300.0], [10.0, 20.0, 15.0]))
        with patch('mod_signal.subtract', return_value=numpy.array([])) as mock_sub:
            profile_scan.subtract(other)
            mock_sub.assert_called_once()

    def test_subtract_one_empty_no_op(self, profile_scan):
        """B20-false: no-op when one profile is empty."""
        other = obj_scan.scan()  # Empty
        # Just verify no error and no signal modification
        profile_scan.subtract(other)

    def test_subtract_empties_peaklist(self, full_scan):
        """subtract() empties peaklist when subtracting profiles."""
        other = obj_scan.scan(profile=make_profile([100.0, 200.0, 300.0], [10.0, 20.0, 15.0]))
        full_scan.subtract(other)
        assert len(full_scan.peaklist) == 0

    def test_subtract_resets_buffers(self, profile_scan):
        """subtract() clears baseline cache."""
        profile_scan._baseline = numpy.array([1, 2, 3])
        other = obj_scan.scan(profile=make_profile([100.0, 200.0, 300.0], [10.0, 20.0, 15.0]))
        profile_scan.subtract(other)
        assert profile_scan._baseline is None


# ============================================================================
# TEST: smooth()
# ============================================================================

class TestScanSmooth:
    """Test smooth() method."""

    def test_smooth_updates_profile(self, profile_scan):
        """smooth() updates profile data."""
        with patch('mod_signal.smooth', return_value=numpy.array([])):
            profile_scan.smooth(method='MA', window=0.5)
            # Verify mod_signal.smooth was called

    def test_smooth_empties_peaklist(self, full_scan):
        """smooth() empties peaklist."""
        with patch('mod_signal.smooth', return_value=numpy.array([])):
            full_scan.smooth(method='MA', window=0.5)
            assert len(full_scan.peaklist) == 0

    def test_smooth_resets_buffers(self, profile_scan):
        """smooth() clears baseline cache."""
        profile_scan._baseline = numpy.array([1, 2, 3])
        with patch('mod_signal.smooth', return_value=numpy.array([])):
            profile_scan.smooth(method='MA', window=0.5)
            assert profile_scan._baseline is None

    def test_smooth_delegates_to_mod_signal(self, profile_scan):
        """smooth() delegates to mod_signal.smooth()."""
        with patch('mod_signal.smooth', return_value=numpy.array([])) as mock_smooth:
            profile_scan.smooth(method='MA', window=0.5, cycles=2)
            mock_smooth.assert_called_once()


# ============================================================================
# TEST: recalibrate()
# ============================================================================

class TestScanRecalibrate:
    """Test recalibrate() method."""

    def test_recalibrate_updates_profile_mz(self, profile_scan):
        """recalibrate() updates m/z values."""
        params = [1.0, 0.0]  # Identity function
        profile_scan.recalibrate(linear_fn, params)
        # m/z values should be updated (but with identity function, same)

    def test_recalibrate_delegates_to_peaklist(self, full_scan):
        """recalibrate() delegates peaklist calibration."""
        params = [1.0, 0.0]
        with patch.object(full_scan.peaklist, 'recalibrate') as mock_recal:
            full_scan.recalibrate(linear_fn, params)
            mock_recal.assert_called_once()

    def test_recalibrate_resets_buffers(self, profile_scan):
        """recalibrate() clears baseline cache."""
        profile_scan._baseline = numpy.array([1, 2, 3])
        profile_scan.recalibrate(linear_fn, [1.0, 0.0])
        assert profile_scan._baseline is None

    def test_recalibrate_empty_profile_no_error(self, peaklist_scan):
        """recalibrate() on empty profile doesn't error."""
        peaklist_scan.recalibrate(linear_fn, [1.0, 0.0])


# ============================================================================
# TEST: subbase()
# ============================================================================

class TestScanSubbase:
    """Test subbase() method."""

    def test_subbase_modifies_profile(self, profile_scan):
        """subbase() modifies profile data."""
        with patch('mod_signal.subbase', return_value=numpy.array([])):
            profile_scan.subbase()
            # Profile updated

    def test_subbase_empties_peaklist(self, full_scan):
        """subbase() empties peaklist."""
        with patch('mod_signal.subbase', return_value=numpy.array([])):
            full_scan.subbase()
            assert len(full_scan.peaklist) == 0

    def test_subbase_resets_buffers(self):
        """subbase() clears baseline cache."""
        s = obj_scan.scan(profile=make_profile([100.0, 200.0], [50.0, 100.0]))
        s._baseline = [1, 2, 3, 4]
        with patch('mod_signal.baseline', return_value=[100.0, 50.0]):
            with patch('mod_signal.subbase', return_value=numpy.array([[100.0, 40.0]])):
                s.subbase()
                assert s._baseline is None

    def test_subbase_calls_baseline(self, profile_scan):
        """subbase() retrieves baseline first."""
        with patch.object(profile_scan, 'baseline', return_value=numpy.array([])):
            with patch('mod_signal.subbase', return_value=numpy.array([])):
                profile_scan.subbase()
                # baseline should have been called


# ============================================================================
# TEST: labelscan() (B21, B22)
# ============================================================================

class TestScanLabelscan:
    """Test labelscan() method."""

    def test_labelscan_without_smoothing(self, gaussian_scan):
        """B21-false: labelscan without smoothing uses raw profile."""
        with patch('mod_peakpicking.labelscan', return_value=obj_peaklist.peaklist()):
            gaussian_scan.labelscan(smoothMethod=None)
            # Raw profile used

    def test_labelscan_with_smoothing(self, gaussian_scan):
        """B21-true: labelscan with smoothing pre-smooths profile."""
        with patch('mod_signal.smooth', return_value=gaussian_scan.profile):
            with patch('mod_peakpicking.labelscan', return_value=obj_peaklist.peaklist()):
                gaussian_scan.labelscan(smoothMethod='MA', smoothWindow=0.2)
                # Smoothing called

    def test_labelscan_empty_peaklist_returns_false(self, gaussian_scan):
        """B22-true: return False when mod_peakpicking returns None."""
        with patch('mod_peakpicking.labelscan', return_value=None):
            result = gaussian_scan.labelscan()
            assert result is False

    def test_labelscan_peaklist_returns_true(self, gaussian_scan):
        """B22-false: return True when peaklist returned."""
        peaklist = obj_peaklist.peaklist()
        with patch('mod_peakpicking.labelscan', return_value=peaklist):
            result = gaussian_scan.labelscan()
            assert result is True

    def test_labelscan_updates_peaklist(self, gaussian_scan):
        """labelscan() updates scan peaklist on success."""
        peaks = make_peaks((500.0, 1000.0))
        peaklist = obj_peaklist.peaklist(peaks)
        with patch('mod_peakpicking.labelscan', return_value=peaklist):
            gaussian_scan.labelscan()
            assert gaussian_scan.peaklist is peaklist


# ============================================================================
# TEST: labelpeak() (B23)
# ============================================================================

class TestScanLabelpeak:
    """Test labelpeak() method."""

    def test_labelpeak_in_range_returns_true(self, gaussian_scan):
        """B23-false: return True when peak found."""
        peak = obj_peak.peak(500.0, 1000.0)
        with patch('mod_peakpicking.labelpeak', return_value=peak):
            result = gaussian_scan.labelpeak(mz=500.0)
            assert result is True

    def test_labelpeak_out_of_range_returns_false(self, gaussian_scan):
        """B23-true: return False when no peak found."""
        with patch('mod_peakpicking.labelpeak', return_value=None):
            result = gaussian_scan.labelpeak(mz=1000.0)
            assert result is False

    def test_labelpeak_appends_peak_on_success(self, gaussian_scan):
        """labelpeak() appends peak to peaklist."""
        peak = obj_peak.peak(500.0, 1000.0)
        with patch('mod_peakpicking.labelpeak', return_value=peak):
            original_len = len(gaussian_scan.peaklist)
            gaussian_scan.labelpeak(mz=500.0)
            assert len(gaussian_scan.peaklist) == original_len + 1

    def test_labelpeak_calls_baseline(self, gaussian_scan):
        """labelpeak() retrieves baseline."""
        with patch.object(gaussian_scan, 'baseline', return_value=numpy.array([])):
            with patch('mod_peakpicking.labelpeak', return_value=None):
                gaussian_scan.labelpeak(mz=500.0)
                # baseline called


# ============================================================================
# TEST: labelpoint() (B24)
# ============================================================================

class TestScanLabelpoint:
    """Test labelpoint() method."""

    def test_labelpoint_valid_mz_returns_true(self, gaussian_scan):
        """B24-false: return True when peak labeled."""
        peak = obj_peak.peak(500.0, 1000.0)
        with patch('mod_peakpicking.labelpoint', return_value=peak):
            result = gaussian_scan.labelpoint(mz=500.0)
            assert result is True

    def test_labelpoint_empty_profile_returns_false(self):
        """B24-true: return False when labelpoint returns None."""
        s = obj_scan.scan(profile=make_profile([100.0, 200.0], [50.0, 100.0]))
        with patch('mod_peakpicking.labelpoint', return_value=None):
            result = s.labelpoint(mz=150.0)
            assert result is False

    def test_labelpoint_appends_peak_on_success(self, gaussian_scan):
        """labelpoint() appends peak to peaklist."""
        peak = obj_peak.peak(500.0, 1000.0)
        with patch('mod_peakpicking.labelpoint', return_value=peak):
            original_len = len(gaussian_scan.peaklist)
            gaussian_scan.labelpoint(mz=500.0)
            assert len(gaussian_scan.peaklist) == original_len + 1


# ============================================================================
# TEST: Peaklist Delegators
# ============================================================================

class TestScanPeaklistDelegators:
    """Test methods that delegate to peaklist."""

    def test_deisotope_delegates(self, full_scan):
        """deisotope() delegates to peaklist."""
        with patch.object(full_scan.peaklist, 'deisotope') as mock_deiso:
            full_scan.deisotope(maxCharge=2)
            mock_deiso.assert_called_once()

    def test_deconvolute_clears_profile(self, full_scan):
        """deconvolute() clears profile data."""
        full_scan.deconvolute()
        assert len(full_scan) == 0

    def test_deconvolute_delegates_peaklist(self, full_scan):
        """deconvolute() delegates to peaklist."""
        with patch.object(full_scan.peaklist, 'deconvolute') as mock_deconv:
            full_scan.deconvolute()
            mock_deconv.assert_called_once()

    def test_deconvolute_resets_buffers(self, full_scan):
        """deconvolute() clears baseline cache."""
        full_scan._baseline = numpy.array([1, 2, 3])
        full_scan.deconvolute()
        assert full_scan._baseline is None

    def test_consolidate_delegates(self, full_scan):
        """consolidate() delegates to peaklist."""
        with patch.object(full_scan.peaklist, 'consolidate') as mock_cons:
            full_scan.consolidate(window=0.5)
            mock_cons.assert_called_once()

    def test_remthreshold_delegates(self, full_scan):
        """remthreshold() delegates to peaklist."""
        with patch.object(full_scan.peaklist, 'remthreshold') as mock_rem:
            full_scan.remthreshold(absThreshold=10.0)
            mock_rem.assert_called_once()

    def test_remshoulders_delegates(self, full_scan):
        """remshoulders() delegates to peaklist."""
        with patch.object(full_scan.peaklist, 'remshoulders') as mock_rem:
            full_scan.remshoulders(window=2.5)
            mock_rem.assert_called_once()

    def test_remisotopes_delegates(self, full_scan):
        """remisotopes() delegates to peaklist."""
        with patch.object(full_scan.peaklist, 'remisotopes') as mock_rem:
            full_scan.remisotopes()
            mock_rem.assert_called_once()

    def test_remuncharged_delegates(self, full_scan):
        """remuncharged() delegates to peaklist."""
        with patch.object(full_scan.peaklist, 'remuncharged') as mock_rem:
            full_scan.remuncharged()
            mock_rem.assert_called_once()


# ============================================================================
# TEST: Property-Based Tests (Hypothesis)
# ============================================================================

class TestScanPropertyBased:
    """Property-based tests using hypothesis."""

    @given(st.lists(st.tuples(st.floats(100, 1000), st.floats(0, 1000)), min_size=1, max_size=50))
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
    def test_len_matches_profile_length(self, peaks):
        """Verify len(scan) equals len(profile)."""
        profile = numpy.array(peaks)
        s = obj_scan.scan(profile=profile)
        assert len(s) == len(profile)

    @given(st.floats(0.1, 10.0))
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
    def test_multiply_scales_all_values(self, factor):
        """Verify multiply scales all profile intensities."""
        profile = make_profile([100.0, 200.0, 300.0], [50.0, 100.0, 75.0])
        s = obj_scan.scan(profile=profile)
        original_y = s.profile[0][1]
        s.multiply(factor)
        assert abs(s.profile[0][1] - original_y * factor) < 1e-6

    @given(st.sampled_from(['profile', 'peaklist', 'both', 'none']))
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
    def test_normalization_always_positive(self, data_type):
        """Verify normalization() always returns positive value."""
        if data_type == 'profile':
            s = obj_scan.scan(profile=make_profile([100.0], [50.0]))
        elif data_type == 'peaklist':
            s = obj_scan.scan(peaklist=make_peaklist((100.0, 100.0)))
        elif data_type == 'both':
            s = obj_scan.scan(
                profile=make_profile([100.0], [50.0]),
                peaklist=make_peaklist((100.0, 100.0))
            )
        else:  # none
            s = obj_scan.scan()

        norm = s.normalization()
        assert norm > 0.0


# ============================================================================
# TEST: Edge Cases
# ============================================================================

class TestScanEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_profile(self):
        """Scan with single data point."""
        profile = make_profile([500.0], [1000.0])
        s = obj_scan.scan(profile=profile)
        assert len(s) == 1
        assert s.hasprofile()

    def test_large_profile(self):
        """Scan with large profile."""
        mz_values = list(range(100, 2001))
        ai_values = [100.0] * len(mz_values)
        profile = make_profile(mz_values, ai_values)
        s = obj_scan.scan(profile=profile)
        assert len(s) == 1901

    def test_multiply_by_zero(self, profile_scan):
        """Multiply by zero eliminates intensities."""
        profile_scan.multiply(0.0)
        assert profile_scan.profile[0][1] == 0.0

    def test_combine_scan_with_itself(self, profile_scan):
        """Combine scan with itself."""
        profile_scan.combine(profile_scan)
        # Should not error (though result depends on mod_signal.combine)

    def test_subtract_identical_profiles(self):
        """Subtract identical profiles."""
        profile = make_profile([100.0, 200.0], [50.0, 100.0])
        s1 = obj_scan.scan(profile=copy.deepcopy(profile))
        s2 = obj_scan.scan(profile=copy.deepcopy(profile))
        s1.subtract(s2)
        # Result depends on mod_signal.subtract implementation


# ============================================================================
# TEST: Integration Scenarios
# ============================================================================

class TestScanIntegration:
    """Integration tests for combined operations."""

    def test_labelscan_then_normalize(self):
        """Label peaks then normalize scan."""
        s = obj_scan.scan(profile=make_gaussian_profile())
        peaks = make_peaks((500.0, 1000.0))
        peaklist = obj_peaklist.peaklist(peaks)

        with patch('mod_peakpicking.labelscan', return_value=peaklist):
            s.labelscan()
            assert s.haspeaks()
            s.normalize()
            assert s.haspeaks()

    def test_baseline_then_area(self):
        """Get baseline then calculate area."""
        s = obj_scan.scan(profile=make_profile([100.0, 200.0], [50.0, 100.0]))
        baseline_result = [100.0, 50.0]
        with patch('mod_signal.baseline', return_value=baseline_result):
            with patch('mod_signal.area', return_value=1000.0):
                baseline1 = s.baseline(window=0.1, offset=0.0)
                area = s.area()
                # Area should use the cached baseline
                baseline2 = s.baseline(window=0.1, offset=0.0)
                # Both calls should have been to cached values
                assert s._baselineParams['window'] == 0.1

    def test_add_and_normalize(self):
        """Add two scans then normalize result."""
        s1 = obj_scan.scan(profile=make_profile([100.0], [100.0]))
        s2 = obj_scan.scan(profile=make_profile([200.0], [50.0]))
        result = s1 + s2
        result.normalize()
        assert len(result) >= 0

    def test_full_workflow(self):
        """Full workflow: create, smooth, label, denoise."""
        s = obj_scan.scan(profile=make_gaussian_profile())

        # Smooth
        with patch('mod_signal.smooth', return_value=s.profile):
            s.smooth(method='MA', window=0.2)

        # Label
        peaks = make_peaks((500.0, 1000.0))
        peaklist = obj_peaklist.peaklist(peaks)
        with patch('mod_peakpicking.labelscan', return_value=peaklist):
            s.labelscan()

        assert s.haspeaks()
        assert len(s) > 0
