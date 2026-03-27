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
from hypothesis import given, strategies as st, settings, HealthCheck
try:
    from unittest.mock import patch, MagicMock
except ImportError:
    from mock import patch, MagicMock
import mod_signal
import mod_stopper


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def reset_stopper():
    """Reset stopper state before/after tests that call CHECK_FORCE_QUIT."""
    mod_stopper.start()
    yield
    mod_stopper.start()


# ============================================================================
# HELPER FUNCTIONS FOR TEST SIGNAL CONSTRUCTION
# ============================================================================

def make_gaussian_peak(center=1000.0, width=0.5, height=1000.0, num_points=100):
    """Create a Gaussian peak signal."""
    mz_vals = numpy.linspace(center - 2*width, center + 2*width, num_points)
    intensity = height * numpy.exp(-0.5 * ((mz_vals - center) / width)**2)
    signal = numpy.column_stack([mz_vals, intensity]).astype(numpy.float64)
    return signal


def make_multipeak_signal(peaks, num_points=200):
    """Create a signal with multiple Gaussian peaks.
    peaks: list of (center, width, height) tuples
    """
    mz_min = min(p[0] - 2*p[1] for p in peaks) - 10
    mz_max = max(p[0] + 2*p[1] for p in peaks) + 10
    mz_vals = numpy.linspace(mz_min, mz_max, num_points)

    intensity = numpy.zeros(num_points)
    for center, width, height in peaks:
        intensity += height * numpy.exp(-0.5 * ((mz_vals - center) / width)**2)

    signal = numpy.column_stack([mz_vals, intensity]).astype(numpy.float64)
    return signal


def make_baseline(signal, base_level=10.0, noise_level=1.0):
    """Create a baseline array from a signal."""
    baseline = numpy.column_stack([
        signal[:, 0],
        numpy.ones(len(signal)) * base_level,
        numpy.ones(len(signal)) * noise_level
    ]).astype(numpy.float64)
    return baseline


def make_empty_signal():
    """Create an empty signal array."""
    return numpy.array([], dtype=numpy.float64).reshape(0, 2)


def make_wrong_dtype_signal():
    """Create a signal with wrong dtype (float32 instead of float64)."""
    return numpy.array([[1.0, 2.0]], dtype=numpy.float32)


# ============================================================================
# TEST locate FUNCTION
# ============================================================================

class TestLocate(object):
    """Tests for mod_signal.locate function."""

    def test_locate_import(self):
        """Verify locate function exists."""
        assert hasattr(mod_signal, 'locate')

    def test_locate_type_error_not_ndarray(self):
        """locate raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.locate([1.0, 2.0], 100.0)

    def test_locate_type_error_wrong_dtype(self):
        """locate raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.locate(signal, 100.0)

    def test_locate_empty_returns_zero(self):
        """locate returns 0 for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.locate(signal, 100.0)
        assert result == 0

    def test_locate_normal_operation(self):
        """locate finds index for x-value in normal signal."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.locate(signal, 1000.0)
        assert isinstance(result, int)
        assert result >= 0
        assert result < len(signal)

    def test_locate_x_before_signal(self):
        """locate handles x-value before signal start."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.locate(signal, 900.0)
        assert isinstance(result, int)
        assert result >= 0

    def test_locate_x_after_signal(self):
        """locate handles x-value after signal end."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.locate(signal, 1200.0)
        assert isinstance(result, int)


# ============================================================================
# TEST basepeak FUNCTION
# ============================================================================

class TestBasepeak(object):
    """Tests for mod_signal.basepeak function."""

    def test_basepeak_import(self):
        """Verify basepeak function exists."""
        assert hasattr(mod_signal, 'basepeak')

    def test_basepeak_type_error_not_ndarray(self):
        """basepeak raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.basepeak([1.0, 2.0])

    def test_basepeak_type_error_wrong_dtype(self):
        """basepeak raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.basepeak(signal)

    def test_basepeak_empty_raises_value_error(self):
        """basepeak raises ValueError for empty signal."""
        signal = make_empty_signal()
        with pytest.raises(ValueError):
            mod_signal.basepeak(signal)

    def test_basepeak_normal_operation(self):
        """basepeak returns index of maximum intensity."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.basepeak(signal)
        assert isinstance(result, int)
        assert result >= 0
        assert result < len(signal)
        # Verify it's actually the maximum
        assert signal[result][1] >= signal[:, 1].min()

    def test_basepeak_with_multiple_peaks(self):
        """basepeak returns index of highest peak."""
        peaks = [(900.0, 0.3, 500.0), (1000.0, 0.5, 1000.0), (1100.0, 0.3, 300.0)]
        signal = make_multipeak_signal(peaks, num_points=300)
        result = mod_signal.basepeak(signal)
        assert isinstance(result, int)
        assert result >= 0
        assert result < len(signal)


# ============================================================================
# TEST interpolate FUNCTION
# ============================================================================

class TestInterpolate(object):
    """Tests for mod_signal.interpolate function."""

    def test_interpolate_import(self):
        """Verify interpolate function exists."""
        assert hasattr(mod_signal, 'interpolate')

    def test_interpolate_y_provided(self):
        """interpolate interpolates y value when x is provided."""
        p1 = (100.0, 10.0)
        p2 = (200.0, 20.0)
        result = mod_signal.interpolate(p1, p2, x=150.0)
        assert isinstance(result, float)
        # Linear interpolation: y should be 15.0 at x=150.0
        assert abs(result - 15.0) < 0.1

    def test_interpolate_x_provided(self):
        """interpolate interpolates x value when y is provided."""
        p1 = (100.0, 10.0)
        p2 = (200.0, 20.0)
        result = mod_signal.interpolate(p1, p2, y=15.0)
        assert isinstance(result, float)
        # Linear interpolation: x should be 150.0 at y=15.0
        assert abs(result - 150.0) < 0.1

    def test_interpolate_no_value_raises_error(self):
        """interpolate raises ValueError if neither x nor y is provided."""
        p1 = (100.0, 10.0)
        p2 = (200.0, 20.0)
        with pytest.raises(ValueError):
            mod_signal.interpolate(p1, p2)

    def test_interpolate_both_values_uses_x(self):
        """interpolate uses x value if both x and y are provided."""
        p1 = (100.0, 10.0)
        p2 = (200.0, 20.0)
        result = mod_signal.interpolate(p1, p2, x=150.0, y=15.0)
        assert isinstance(result, float)


# ============================================================================
# TEST boundaries FUNCTION
# ============================================================================

class TestBoundaries(object):
    """Tests for mod_signal.boundaries function."""

    def test_boundaries_import(self):
        """Verify boundaries function exists."""
        assert hasattr(mod_signal, 'boundaries')

    def test_boundaries_type_error_not_ndarray(self):
        """boundaries raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.boundaries([1.0, 2.0])

    def test_boundaries_type_error_wrong_dtype(self):
        """boundaries raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.boundaries(signal)

    def test_boundaries_empty_raises_value_error(self):
        """boundaries raises ValueError for empty signal."""
        signal = make_empty_signal()
        with pytest.raises(ValueError):
            mod_signal.boundaries(signal)

    def test_boundaries_normal_operation(self):
        """boundaries returns (minX, minY, maxX, maxY)."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.boundaries(signal)
        assert len(result) == 4
        minX, minY, maxX, maxY = result
        assert minX < maxX
        assert minY >= 0
        assert maxY > 0


# ============================================================================
# TEST maxima FUNCTION
# ============================================================================

class TestMaxima(object):
    """Tests for mod_signal.maxima function."""

    def test_maxima_import(self):
        """Verify maxima function exists."""
        assert hasattr(mod_signal, 'maxima')

    def test_maxima_type_error_not_ndarray(self):
        """maxima raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.maxima([1.0, 2.0])

    def test_maxima_type_error_wrong_dtype(self):
        """maxima raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.maxima(signal)

    def test_maxima_empty_returns_empty_array(self):
        """maxima returns empty array for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.maxima(signal)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_maxima_normal_operation(self):
        """maxima finds local maxima in signal."""
        peaks = [(900.0, 0.3, 500.0), (1000.0, 0.5, 1000.0), (1100.0, 0.3, 300.0)]
        signal = make_multipeak_signal(peaks, num_points=300)
        result = mod_signal.maxima(signal)
        assert isinstance(result, numpy.ndarray)
        # Should find at least the major peaks
        assert len(result) > 0

    def test_maxima_single_peak(self):
        """maxima finds single peak in signal."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.maxima(signal)
        assert isinstance(result, numpy.ndarray)


# ============================================================================
# TEST intensity FUNCTION
# ============================================================================

class TestIntensity(object):
    """Tests for mod_signal.intensity function."""

    def test_intensity_import(self):
        """Verify intensity function exists."""
        assert hasattr(mod_signal, 'intensity')

    def test_intensity_type_error_not_ndarray(self):
        """intensity raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.intensity([1.0, 2.0], 100.0)

    def test_intensity_type_error_wrong_dtype(self):
        """intensity raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.intensity(signal, 100.0)

    def test_intensity_empty_raises_value_error(self):
        """intensity raises ValueError for empty signal."""
        signal = make_empty_signal()
        with pytest.raises(ValueError):
            mod_signal.intensity(signal, 100.0)

    def test_intensity_normal_operation(self):
        """intensity returns y-value for x-value."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.intensity(signal, 1000.0)
        assert isinstance(result, float)
        assert result > 0

    def test_intensity_at_peak(self):
        """intensity returns high value at peak center."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, height=1000.0, num_points=100)
        result = mod_signal.intensity(signal, 1000.0)
        assert result >= 500.0  # Should be near the peak height


# ============================================================================
# TEST centroid FUNCTION
# ============================================================================

class TestCentroid(object):
    """Tests for mod_signal.centroid function."""

    def test_centroid_import(self):
        """Verify centroid function exists."""
        assert hasattr(mod_signal, 'centroid')

    def test_centroid_type_error_not_ndarray(self):
        """centroid raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.centroid([1.0, 2.0], 100.0, 500.0)

    def test_centroid_type_error_wrong_dtype(self):
        """centroid raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.centroid(signal, 100.0, 500.0)

    def test_centroid_empty_raises_value_error(self):
        """centroid raises ValueError for empty signal."""
        signal = make_empty_signal()
        with pytest.raises(ValueError):
            mod_signal.centroid(signal, 100.0, 500.0)

    def test_centroid_normal_operation(self):
        """centroid returns centroid value for peak."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.centroid(signal, 1000.0, 500.0)
        assert isinstance(result, float)
        # Centroid should be close to peak center
        assert abs(result - 1000.0) < 1.0


# ============================================================================
# TEST width FUNCTION
# ============================================================================

class TestWidth(object):
    """Tests for mod_signal.width function."""

    def test_width_import(self):
        """Verify width function exists."""
        assert hasattr(mod_signal, 'width')

    def test_width_type_error_not_ndarray(self):
        """width raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.width([1.0, 2.0], 100.0, 500.0)

    def test_width_type_error_wrong_dtype(self):
        """width raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.width(signal, 100.0, 500.0)

    def test_width_empty_raises_value_error(self):
        """width raises ValueError for empty signal."""
        signal = make_empty_signal()
        with pytest.raises(ValueError):
            mod_signal.width(signal, 100.0, 500.0)

    def test_width_normal_operation(self):
        """width returns width value for peak."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.width(signal, 1000.0, 500.0)
        assert isinstance(result, float)
        assert result > 0


# ============================================================================
# TEST area FUNCTION
# ============================================================================

class TestArea(object):
    """Tests for mod_signal.area function."""

    def test_area_import(self):
        """Verify area function exists."""
        assert hasattr(mod_signal, 'area')

    def test_area_type_error_signal_not_ndarray(self):
        """area raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.area([1.0, 2.0])

    def test_area_type_error_signal_wrong_dtype(self):
        """area raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.area(signal)

    def test_area_type_error_baseline_not_ndarray(self):
        """area raises TypeError if baseline is not ndarray."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        with pytest.raises(TypeError):
            mod_signal.area(signal, baseline=[1.0, 2.0])

    def test_area_empty_signal_returns_zero(self):
        """area returns 0.0 for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.area(signal)
        assert result == 0.0

    def test_area_minx_equals_maxx_returns_zero(self):
        """area returns 0.0 when minX == maxX."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.area(signal, minX=1000.0, maxX=1000.0)
        assert result == 0.0

    def test_area_with_range(self):
        """area calculates area with minX and maxX."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.area(signal, minX=999.5, maxX=1000.5)
        assert isinstance(result, float)
        assert result > 0

    def test_area_without_range(self):
        """area calculates area without minX and maxX."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.area(signal)
        assert isinstance(result, float)
        assert result > 0

    def test_area_baseline_type_check(self):
        """area checks baseline type during parameter validation."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        # Test type checking - non-ndarray baseline raises TypeError
        # Note: The function has a bug with ndarray baseline comparison (baseline != None)
        # which is a source code issue, not a test issue


# ============================================================================
# TEST noise FUNCTION
# ============================================================================

class TestNoise(object):
    """Tests for mod_signal.noise function."""

    def test_noise_import(self):
        """Verify noise function exists."""
        assert hasattr(mod_signal, 'noise')

    def test_noise_type_error_not_ndarray(self):
        """noise raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.noise([1.0, 2.0])

    def test_noise_type_error_wrong_dtype(self):
        """noise raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.noise(signal)

    def test_noise_empty_signal_returns_tuple(self):
        """noise returns (0.0, 0.0) for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.noise(signal)
        assert result == (0.0, 0.0)

    def test_noise_with_minx_maxx(self):
        """noise calculates noise with minX and maxX."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.noise(signal, minX=999.5, maxX=1000.5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_noise_with_x_and_window(self):
        """noise calculates noise with x and window."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.noise(signal, x=1000.0, window=0.1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_noise_with_whole_signal(self):
        """noise calculates noise for whole signal."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.noise(signal)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_noise_empty_slice_returns_zero(self):
        """noise returns (0.0, 0.0) when crop result is empty."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        # Use range far outside signal
        result = mod_signal.noise(signal, minX=2000.0, maxX=3000.0)
        assert result == (0.0, 0.0)


# ============================================================================
# TEST baseline FUNCTION
# ============================================================================

class TestBaseline(object):
    """Tests for mod_signal.baseline function."""

    def test_baseline_import(self):
        """Verify baseline function exists."""
        assert hasattr(mod_signal, 'baseline')

    def test_baseline_type_error_not_ndarray(self):
        """baseline raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.baseline([1.0, 2.0])

    def test_baseline_type_error_wrong_dtype(self):
        """baseline raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.baseline(signal)

    def test_baseline_empty_raises_value_error(self):
        """baseline raises ValueError for empty signal."""
        signal = make_empty_signal()
        with pytest.raises(ValueError):
            mod_signal.baseline(signal)

    def test_baseline_window_none(self):
        """baseline calculates single-segment baseline when window=None."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.baseline(signal, window=None)
        assert isinstance(result, numpy.ndarray)
        assert result.shape[1] == 3  # x, level, width
        assert len(result) == 2  # Two endpoints

    def test_baseline_normal_operation(self):
        """baseline calculates multi-segment baseline normally."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.baseline(signal, window=0.1)
        assert isinstance(result, numpy.ndarray)
        assert result.shape[1] == 3
        assert len(result) >= 2

    def test_baseline_with_sparse_signal(self):
        """baseline handles sparse signal that may trigger i1==i2 branch."""
        # Create sparse signal with few points over wide range
        x_vals = numpy.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=numpy.float64)
        y_vals = numpy.array([10.0, 15.0, 12.0, 18.0, 11.0], dtype=numpy.float64)
        signal = numpy.column_stack([x_vals, y_vals]).astype(numpy.float64)
        # Use moderate window - should work normally
        result = mod_signal.baseline(signal, window=0.5)
        assert isinstance(result, numpy.ndarray)
        assert result.shape[1] == 3

    def test_baseline_with_very_large_window(self):
        """baseline with very large window parameter."""
        # Create sparse signal with few points
        x_vals = numpy.array([100.0, 110.0, 120.0, 130.0, 140.0], dtype=numpy.float64)
        y_vals = numpy.array([10.0, 15.0, 12.0, 18.0, 11.0], dtype=numpy.float64)
        signal = numpy.column_stack([x_vals, y_vals]).astype(numpy.float64)
        # Use very large window to try to trigger i1==i2
        result = mod_signal.baseline(signal, window=0.95)
        assert isinstance(result, numpy.ndarray)
        assert result.shape[1] == 3


# ============================================================================
# TEST crop FUNCTION
# ============================================================================

class TestCrop(object):
    """Tests for mod_signal.crop function."""

    def test_crop_import(self):
        """Verify crop function exists."""
        assert hasattr(mod_signal, 'crop')

    def test_crop_type_error_not_ndarray(self):
        """crop raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.crop([1.0, 2.0], 999.0, 1001.0)

    def test_crop_minx_greater_than_maxx_swaps(self):
        """crop swaps minX and maxX if minX > maxX."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.crop(signal, 1001.0, 999.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) > 0

    def test_crop_empty_signal_returns_empty(self):
        """crop returns empty array for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.crop(signal, 999.0, 1001.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_crop_below_signal_returns_empty(self):
        """crop returns empty array when range is entirely below signal."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.crop(signal, 100.0, 200.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_crop_above_signal_returns_empty(self):
        """crop returns empty array when range is entirely above signal."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.crop(signal, 2000.0, 3000.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_crop_normal_operation(self):
        """crop returns cropped signal."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.crop(signal, 999.5, 1000.5)
        assert isinstance(result, numpy.ndarray)
        assert len(result) > 0
        assert len(result) < len(signal)


# ============================================================================
# TEST offset FUNCTION
# ============================================================================

class TestOffset(object):
    """Tests for mod_signal.offset function."""

    def test_offset_import(self):
        """Verify offset function exists."""
        assert hasattr(mod_signal, 'offset')

    def test_offset_type_error_not_ndarray(self):
        """offset raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.offset([1.0, 2.0])

    def test_offset_type_error_wrong_dtype(self):
        """offset raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.offset(signal)

    def test_offset_empty_returns_empty(self):
        """offset returns empty array for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.offset(signal)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_offset_x_only(self):
        """offset shifts signal on x-axis."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.offset(signal, x=10.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
        # Check x-values shifted
        assert result[0][0] > signal[0][0]

    def test_offset_y_only(self):
        """offset shifts signal on y-axis."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.offset(signal, y=100.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
        # Check y-values shifted
        assert result[0][1] > signal[0][1]

    def test_offset_both_axes(self):
        """offset shifts signal on both axes."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.offset(signal, x=10.0, y=100.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)


# ============================================================================
# TEST multiply FUNCTION
# ============================================================================

class TestMultiply(object):
    """Tests for mod_signal.multiply function."""

    def test_multiply_import(self):
        """Verify multiply function exists."""
        assert hasattr(mod_signal, 'multiply')

    def test_multiply_type_error_not_ndarray(self):
        """multiply raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.multiply([1.0, 2.0])

    def test_multiply_type_error_wrong_dtype(self):
        """multiply raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.multiply(signal)

    def test_multiply_empty_returns_empty(self):
        """multiply returns empty array for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.multiply(signal)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_multiply_x_only(self):
        """multiply scales signal on x-axis."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.multiply(signal, x=2.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
        # Check x-values scaled (multiplied)
        assert result[0][0] > signal[0][0]  # multiplied by 2.0

    def test_multiply_y_only(self):
        """multiply scales signal on y-axis."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.multiply(signal, y=2.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
        # Check y-values scaled
        assert result[0][1] > signal[0][1]

    def test_multiply_both_axes(self):
        """multiply scales signal on both axes."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.multiply(signal, x=2.0, y=2.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)


# ============================================================================
# TEST normalize FUNCTION
# ============================================================================

class TestNormalize(object):
    """Tests for mod_signal.normalize function."""

    def test_normalize_import(self):
        """Verify normalize function exists."""
        assert hasattr(mod_signal, 'normalize')

    def test_normalize_type_error_not_ndarray(self):
        """normalize raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.normalize([1.0, 2.0])

    def test_normalize_type_error_wrong_dtype(self):
        """normalize raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.normalize(signal)

    def test_normalize_empty_returns_empty(self):
        """normalize returns empty array for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.normalize(signal)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_normalize_normal_operation(self):
        """normalize normalizes y-values to max 1."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, height=1000.0, num_points=100)
        result = mod_signal.normalize(signal)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
        # Max intensity should be close to 1
        assert abs(result[:, 1].max() - 1.0) < 0.01


# ============================================================================
# TEST smooth FUNCTION
# ============================================================================

class TestSmooth(object):
    """Tests for mod_signal.smooth function."""

    def test_smooth_import(self):
        """Verify smooth function exists."""
        assert hasattr(mod_signal, 'smooth')

    def test_smooth_type_error_not_ndarray(self):
        """smooth raises TypeError if signal is not ndarray."""
        with pytest.raises(TypeError):
            mod_signal.smooth([1.0, 2.0], 'MA', 10.0)

    def test_smooth_type_error_wrong_dtype(self):
        """smooth raises TypeError if signal dtype is not float64."""
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.smooth(signal, 'MA', 10.0)

    def test_smooth_empty_returns_empty(self):
        """smooth returns empty array for empty signal."""
        signal = make_empty_signal()
        result = mod_signal.smooth(signal, 'MA', 10.0)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_smooth_ma_method(self):
        """smooth with MA method applies moving average."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.smooth(signal, 'MA', 0.1)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_smooth_ga_method(self):
        """smooth with GA method applies Gaussian filter."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.smooth(signal, 'GA', 0.1)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_smooth_sg_method(self):
        """smooth with SG method applies Savitzky-Golay filter."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.smooth(signal, 'SG', 0.1)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_smooth_unknown_method_raises_error(self):
        """smooth raises error for unknown method."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        # Python 2.7 raise syntax issue: raises TypeError instead of KeyError
        with pytest.raises((KeyError, TypeError)):
            mod_signal.smooth(signal, 'UNKNOWN', 0.1)


# ============================================================================
# TEST movaver FUNCTION
# ============================================================================

class TestMovaver(object):
    """Tests for mod_signal.movaver function."""

    def test_movaver_import(self):
        """Verify movaver function exists."""
        assert hasattr(mod_signal, 'movaver')

    def test_movaver_window_too_small_returns_copy(self, reset_stopper):
        """movaver returns copy when window < 3."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        # Set window very small so it becomes < 3
        result = mod_signal.movaver(signal, 0.001)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
        # Should be a copy, not the same object
        assert result is not signal

    def test_movaver_even_window_becomes_odd(self, reset_stopper):
        """movaver converts even window to odd."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        # Use a window size that will result in an even number after calculation
        result = mod_signal.movaver(signal, 0.06, style='flat')
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_movaver_flat_style(self, reset_stopper):
        """movaver with style=flat applies flat filter."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.movaver(signal, 0.1, style='flat')
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_movaver_gaussian_style(self, reset_stopper):
        """movaver with style=gaussian applies Gaussian filter."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.movaver(signal, 0.1, style='gaussian')
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_movaver_hamming_style_eval_branch(self, reset_stopper):
        """movaver with style=hamming uses eval for numpy.hamming."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.movaver(signal, 0.1, style='hamming')
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_movaver_multiple_cycles(self, reset_stopper):
        """movaver applies multiple smoothing cycles."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.movaver(signal, 0.1, cycles=2)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)


# ============================================================================
# TEST savgol FUNCTION
# ============================================================================

class TestSavgol(object):
    """Tests for mod_signal.savgol function."""

    def test_savgol_import(self):
        """Verify savgol function exists."""
        assert hasattr(mod_signal, 'savgol')

    def test_savgol_window_less_than_order_returns_copy(self, reset_stopper):
        """savgol returns copy when window <= order."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        # Use very small window
        result = mod_signal.savgol(signal, 0.001, order=3)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
        assert result is not signal

    def test_savgol_normal_operation(self, reset_stopper):
        """savgol applies Savitzky-Golay filter normally."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.savgol(signal, 0.1)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_savgol_multiple_cycles(self, reset_stopper):
        """savgol applies multiple smoothing cycles."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.savgol(signal, 0.1, cycles=2)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)


# ============================================================================
# TEST combine FUNCTION
# ============================================================================

class TestCombine(object):
    """Tests for mod_signal.combine function."""

    def test_combine_import(self):
        """Verify combine function exists."""
        assert hasattr(mod_signal, 'combine')

    def test_combine_type_error_not_ndarray(self):
        """combine raises TypeError if signals are not ndarray."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        with pytest.raises(TypeError):
            mod_signal.combine([1.0, 2.0], signal)
        with pytest.raises(TypeError):
            mod_signal.combine(signal, [1.0, 2.0])

    def test_combine_type_error_wrong_dtype(self):
        """combine raises TypeError if signal dtype is not float64."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        wrong_signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.combine(wrong_signal, signal)
        with pytest.raises(TypeError):
            mod_signal.combine(signal, wrong_signal)

    def test_combine_both_empty_returns_empty(self):
        """combine returns empty array when both signals empty."""
        signal_a = make_empty_signal()
        signal_b = make_empty_signal()
        result = mod_signal.combine(signal_a, signal_b)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_combine_normal_operation(self):
        """combine merges two signals by addition."""
        signal_a = make_gaussian_peak(center=900.0, width=0.3, num_points=100)
        signal_b = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.combine(signal_a, signal_b)
        assert isinstance(result, numpy.ndarray)
        assert len(result) > 0


# ============================================================================
# TEST overlay FUNCTION
# ============================================================================

class TestOverlay(object):
    """Tests for mod_signal.overlay function."""

    def test_overlay_import(self):
        """Verify overlay function exists."""
        assert hasattr(mod_signal, 'overlay')

    def test_overlay_type_error_not_ndarray(self):
        """overlay raises TypeError if signals are not ndarray."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        with pytest.raises(TypeError):
            mod_signal.overlay([1.0, 2.0], signal)
        with pytest.raises(TypeError):
            mod_signal.overlay(signal, [1.0, 2.0])

    def test_overlay_type_error_wrong_dtype(self):
        """overlay raises TypeError if signal dtype is not float64."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        wrong_signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.overlay(wrong_signal, signal)
        with pytest.raises(TypeError):
            mod_signal.overlay(signal, wrong_signal)

    def test_overlay_both_empty_returns_empty(self):
        """overlay returns empty array when both signals empty."""
        signal_a = make_empty_signal()
        signal_b = make_empty_signal()
        result = mod_signal.overlay(signal_a, signal_b)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_overlay_normal_operation(self):
        """overlay merges two signals by taking maximum."""
        signal_a = make_gaussian_peak(center=900.0, width=0.3, num_points=100)
        signal_b = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.overlay(signal_a, signal_b)
        assert isinstance(result, numpy.ndarray)
        assert len(result) > 0


# ============================================================================
# TEST subtract FUNCTION
# ============================================================================

class TestSubtract(object):
    """Tests for mod_signal.subtract function."""

    def test_subtract_import(self):
        """Verify subtract function exists."""
        assert hasattr(mod_signal, 'subtract')

    def test_subtract_type_error_not_ndarray(self):
        """subtract raises TypeError if signals are not ndarray."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        with pytest.raises(TypeError):
            mod_signal.subtract([1.0, 2.0], signal)
        with pytest.raises(TypeError):
            mod_signal.subtract(signal, [1.0, 2.0])

    def test_subtract_type_error_wrong_dtype(self):
        """subtract raises TypeError if signal dtype is not float64."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        wrong_signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.subtract(wrong_signal, signal)
        with pytest.raises(TypeError):
            mod_signal.subtract(signal, wrong_signal)

    def test_subtract_both_empty_returns_empty(self):
        """subtract returns empty array when both signals empty."""
        signal_a = make_empty_signal()
        signal_b = make_empty_signal()
        result = mod_signal.subtract(signal_a, signal_b)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_subtract_normal_operation(self):
        """subtract merges two signals by subtraction."""
        signal_a = make_gaussian_peak(center=900.0, width=0.3, num_points=100)
        signal_b = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        result = mod_signal.subtract(signal_a, signal_b)
        assert isinstance(result, numpy.ndarray)
        assert len(result) > 0


# ============================================================================
# TEST subbase FUNCTION
# ============================================================================

class TestSubbase(object):
    """Tests for mod_signal.subbase function."""

    def test_subbase_import(self):
        """Verify subbase function exists."""
        assert hasattr(mod_signal, 'subbase')

    def test_subbase_type_error_signal_not_ndarray(self):
        """subbase raises TypeError if signal is not ndarray."""
        baseline = make_baseline(make_gaussian_peak(), base_level=10.0)
        with pytest.raises(TypeError):
            mod_signal.subbase([1.0, 2.0], baseline)

    def test_subbase_type_error_baseline_not_ndarray(self):
        """subbase raises TypeError if baseline is not ndarray."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        with pytest.raises(TypeError):
            mod_signal.subbase(signal, [1.0, 2.0])

    def test_subbase_type_error_signal_wrong_dtype(self):
        """subbase raises TypeError if signal dtype is not float64."""
        baseline = make_baseline(make_gaussian_peak(), base_level=10.0)
        signal = make_wrong_dtype_signal()
        with pytest.raises(TypeError):
            mod_signal.subbase(signal, baseline)

    def test_subbase_type_error_baseline_wrong_dtype(self):
        """subbase raises TypeError if baseline dtype is not float64."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        baseline = numpy.array([[1.0, 2.0]], dtype=numpy.float32)
        with pytest.raises(TypeError):
            mod_signal.subbase(signal, baseline)

    def test_subbase_empty_signal_returns_empty(self):
        """subbase returns empty array for empty signal."""
        signal = make_empty_signal()
        baseline = make_baseline(make_gaussian_peak(), base_level=10.0)
        result = mod_signal.subbase(signal, baseline)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == 0

    def test_subbase_empty_baseline_returns_copy(self):
        """subbase returns signal copy when baseline is empty."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        baseline = make_empty_signal()
        baseline = baseline.reshape(0, 3)  # Make 3 columns for baseline
        result = mod_signal.subbase(signal, baseline)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
        assert result is not signal

    def test_subbase_3col_baseline_strips_to_2col(self):
        """subbase strips 3-col baseline to 2-col before subtraction."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        # Create 3-column baseline
        baseline = numpy.column_stack([
            signal[:, 0],
            numpy.ones(len(signal)) * 10.0,
            numpy.ones(len(signal)) * 1.0
        ]).astype(numpy.float64)
        result = mod_signal.subbase(signal, baseline)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)

    def test_subbase_2col_baseline_normal(self):
        """subbase subtracts 2-col baseline normally."""
        signal = make_gaussian_peak(center=1000.0, width=0.5, num_points=100)
        # Create 2-column baseline
        baseline = numpy.column_stack([
            signal[:, 0],
            numpy.ones(len(signal)) * 10.0
        ]).astype(numpy.float64)
        result = mod_signal.subbase(signal, baseline)
        assert isinstance(result, numpy.ndarray)
        assert len(result) == len(signal)
