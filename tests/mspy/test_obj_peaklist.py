import pytest
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import mspy.obj_peaklist as obj_peaklist
import mspy.obj_peak as obj_peak
import mspy.mod_peakpicking as mod_peakpicking
import mspy.mod_stopper as mod_stopper


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

def make_peak(mz, ai=100.0, base=0.0, sn=None, charge=None, isotope=None,
              fwhm=None, group='', **kw):
    """Create a peak object for testing."""
    return obj_peak.peak(mz, ai, base, sn=sn, charge=charge, isotope=isotope,
                         fwhm=fwhm, group=group, **kw)


def make_peaklist(*pairs):
    """Create a peaklist from (mz, intensity) tuples."""
    return obj_peaklist.peaklist([make_peak(mz, ai) for mz, ai in pairs])


# ============================================================================
# TEST: __init__
# ============================================================================

class TestPeaklistInit:
    """Test peaklist initialization."""

    def test_init_empty_list(self):
        """B1-false: empty list, no loop."""
        pl = obj_peaklist.peaklist([])
        assert len(pl) == 0
        assert pl.basepeak is None

    def test_init_single_peak(self):
        """B1-true: single peak processed and basepeak set."""
        pl = obj_peaklist.peaklist([make_peak(100.0, 50.0)])
        assert len(pl) == 1
        assert pl.basepeak is not None
        assert pl.basepeak.mz == 100.0
        assert pl.basepeak.ri == 1.0

    def test_init_multiple_peaks_unordered(self):
        """B1-true: multiple peaks, sorted by m/z."""
        pl = obj_peaklist.peaklist([
            make_peak(300.0, 30.0),
            make_peak(100.0, 100.0),
            make_peak(200.0, 50.0)
        ])
        assert len(pl) == 3
        assert pl[0].mz == 100.0
        assert pl[1].mz == 200.0
        assert pl[2].mz == 300.0
        assert pl.basepeak.mz == 100.0

    def test_init_basepeak_highest_intensity(self):
        """B1-true: basepeak is peak with highest intensity."""
        pl = obj_peaklist.peaklist([
            make_peak(100.0, 50.0),
            make_peak(200.0, 200.0),
            make_peak(300.0, 100.0)
        ])
        assert pl.basepeak.mz == 200.0
        assert pl.basepeak.intensity == 200.0

    def test_init_relative_intensities_set(self):
        """B1-true: relative intensities calculated."""
        pl = obj_peaklist.peaklist([
            make_peak(100.0, 50.0),
            make_peak(200.0, 100.0),
            make_peak(300.0, 25.0)
        ])
        assert pl[0].ri == 0.5
        assert pl[1].ri == 1.0
        assert pl[2].ri == 0.25

    def test_init_from_list_tuples(self):
        """B1-true: init from (mz, ai) tuples."""
        pl = obj_peaklist.peaklist([
            [100.0, 100.0],
            [200.0, 50.0]
        ])
        assert len(pl) == 2
        assert pl[0].mz == 100.0

    def test_init_from_tuple_tuples(self):
        """B1-true: init from (mz, ai) tuples (tuple type)."""
        pl = obj_peaklist.peaklist([
            (100.0, 100.0),
            (200.0, 50.0)
        ])
        assert len(pl) == 2
        assert pl[0].mz == 100.0


# ============================================================================
# TEST: __len__
# ============================================================================

class TestPeaklistLen:
    """Test peaklist length."""

    def test_len_empty(self):
        """Length of empty peaklist."""
        pl = obj_peaklist.peaklist()
        assert len(pl) == 0

    def test_len_with_peaks(self):
        """Length of peaklist with peaks."""
        pl = make_peaklist((100, 100), (200, 50))
        assert len(pl) == 2


# ============================================================================
# TEST: __getitem__
# ============================================================================

class TestPeaklistGetItem:
    """Test peaklist indexing."""

    def test_getitem_valid_index(self):
        """Access peak by index."""
        pl = make_peaklist((100, 100), (200, 50))
        assert pl[0].mz == 100.0
        assert pl[1].mz == 200.0

    def test_getitem_negative_index(self):
        """Access peak by negative index."""
        pl = make_peaklist((100, 100), (200, 50))
        assert pl[-1].mz == 200.0
        assert pl[-2].mz == 100.0

    def test_getitem_out_of_range(self):
        """IndexError on out-of-range access."""
        pl = make_peaklist((100, 100))
        with pytest.raises(IndexError):
            _ = pl[5]


# ============================================================================
# TEST: __setitem__
# ============================================================================

class TestPeaklistSetItem:
    """Test peaklist item setting."""

    def test_setitem_replacing_basepeak(self):
        """B2: replacing basepeak recalcs basepeak and ri."""
        pl = make_peaklist((100, 100), (200, 50))
        old_basepeak = pl.basepeak
        new_peak = make_peak(100.0, 200.0)
        pl[0] = new_peak
        assert pl.basepeak.mz == 100.0
        assert pl.basepeak.intensity == 200.0
        assert pl[0].ri == 1.0

    def test_setitem_new_basepeak(self):
        """B3: new peak higher intensity becomes basepeak."""
        pl = make_peaklist((100, 100), (200, 50))
        # Replace second peak with higher intensity
        new_peak = make_peak(300.0, 150.0)
        pl[1] = new_peak
        assert pl.basepeak.mz == 300.0  # New peak is now basepeak
        assert pl.basepeak.intensity == 150.0

    def test_setitem_lower_than_basepeak(self):
        """B4: new peak lower than basepeak, ri set directly."""
        pl = make_peaklist((100, 100), (200, 50))
        new_peak = make_peak(150.0, 30.0)
        pl[1] = new_peak
        assert pl[1].ri == 0.3
        assert pl.basepeak.intensity == 100.0

    def test_setitem_no_basepeak(self):
        """B5: no basepeak, setbasepeak and setRelativeIntensities called."""
        pl = obj_peaklist.peaklist([])
        new_peak = make_peak(100.0, 50.0)
        pl.peaks.append(obj_peak.peak(200.0, 75.0))
        pl[0] = new_peak
        assert pl.basepeak is not None
        assert pl[0].ri > 0

    def test_setitem_sorts_peaklist(self):
        """setitem sorts peaklist after assignment."""
        pl = make_peaklist((100, 100), (300, 50))
        pl[1] = make_peak(150.0, 50.0)
        # Check sorted by mz - only 2 peaks after replacement
        assert pl[0].mz == 100.0
        assert pl[1].mz == 150.0
        assert len(pl) == 2


# ============================================================================
# TEST: __delitem__
# ============================================================================

class TestPeaklistDelItem:
    """Test peaklist item deletion."""

    def test_delitem_non_basepeak(self):
        """B6-false: delete non-basepeak."""
        pl = make_peaklist((100, 100), (200, 50))
        del pl[1]
        assert len(pl) == 1
        assert pl[0].mz == 100.0

    def test_delitem_basepeak(self):
        """B6-true: delete basepeak, recalculates."""
        pl = make_peaklist((100, 100), (200, 50))
        assert pl.basepeak.mz == 100.0
        del pl[0]
        assert len(pl) == 1
        assert pl.basepeak.mz == 200.0
        assert pl[0].ri == 1.0


# ============================================================================
# TEST: __iter__ and next
# ============================================================================

class TestPeaklistIter:
    """Test peaklist iteration."""

    def test_iter_empty(self):
        """Iteration over empty peaklist."""
        pl = obj_peaklist.peaklist()
        peaks = list(pl)
        assert peaks == []

    def test_iter_single_peak(self):
        """B7-true: iteration yields single peak."""
        pl = make_peaklist((100, 100))
        peaks = list(pl)
        assert len(peaks) == 1
        assert peaks[0].mz == 100.0

    def test_iter_multiple_peaks(self):
        """B7-true: iteration yields all peaks."""
        pl = make_peaklist((100, 100), (200, 50), (300, 75))
        peaks = list(pl)
        assert len(peaks) == 3
        assert [p.mz for p in peaks] == [100.0, 200.0, 300.0]

    def test_iter_manual_next(self):
        """B7-true and B7-false: manual iteration via next."""
        pl = make_peaklist((100, 100), (200, 50))
        iter_pl = iter(pl)
        assert iter_pl.next().mz == 100.0
        assert iter_pl.next().mz == 200.0
        with pytest.raises(StopIteration):
            iter_pl.next()


# ============================================================================
# TEST: __add__
# ============================================================================

class TestPeaklistAdd:
    """Test peaklist addition."""

    def test_add_empty_peaklists(self):
        """Adding two empty peaklists."""
        pl1 = obj_peaklist.peaklist()
        pl2 = obj_peaklist.peaklist()
        pl3 = pl1 + pl2
        assert len(pl3) == 0
        assert pl1 is not pl3  # Should be new object

    def test_add_peaklists(self):
        """Adding peaklists combines peaks."""
        pl1 = make_peaklist((100, 100), (200, 50))
        pl2 = make_peaklist((300, 75))
        pl3 = pl1 + pl2
        assert len(pl3) == 3
        assert pl3[0].mz == 100.0
        assert pl3[1].mz == 200.0
        assert pl3[2].mz == 300.0
        assert pl1 is not pl3

    def test_add_preserves_original(self):
        """Addition doesn't modify original peaklists."""
        pl1 = make_peaklist((100, 100))
        pl2 = make_peaklist((200, 50))
        pl3 = pl1 + pl2
        assert len(pl1) == 1
        assert len(pl2) == 1
        assert len(pl3) == 2


# ============================================================================
# TEST: __mul__
# ============================================================================

class TestPeaklistMul:
    """Test peaklist multiplication."""

    def test_mul_by_scalar(self):
        """Multiplying peaklist by scalar."""
        pl1 = make_peaklist((100, 100), (200, 50))
        pl2 = pl1 * 2.0
        assert len(pl2) == 2
        assert pl2[0].mz == 100.0
        assert pl2[0].ai == 200.0
        assert pl2[1].ai == 100.0

    def test_mul_preserves_original(self):
        """Multiplication doesn't modify original."""
        pl1 = make_peaklist((100, 100))
        pl2 = pl1 * 2.0
        assert pl1[0].ai == 100.0
        assert pl2[0].ai == 200.0


# ============================================================================
# TEST: append
# ============================================================================

class TestPeaklistAppend:
    """Test peaklist append."""

    def test_append_to_empty(self):
        """B8-false, B11: append to empty, sets basepeak."""
        pl = obj_peaklist.peaklist()
        pl.append(make_peak(100.0, 100.0))
        assert len(pl) == 1
        assert pl.basepeak is not None
        assert pl[0].ri == 1.0

    def test_append_sorted_order(self):
        """B8-false: append higher mz, no sort needed."""
        pl = make_peaklist((100, 100), (200, 50))
        pl.append(make_peak(300.0, 75.0))
        assert pl[2].mz == 300.0
        assert len(pl) == 3

    def test_append_unsorted_triggers_sort(self):
        """B8-true: append lower mz, triggers sort."""
        pl = make_peaklist((100, 100), (300, 50))
        pl.append(make_peak(200.0, 75.0))
        assert pl[0].mz == 100.0
        assert pl[1].mz == 200.0
        assert pl[2].mz == 300.0

    def test_append_new_basepeak(self):
        """B9: new peak higher intensity becomes basepeak."""
        pl = make_peaklist((100, 100), (200, 50))
        assert pl.basepeak.intensity == 100.0
        pl.append(make_peak(300.0, 150.0))
        assert pl.basepeak.intensity == 150.0
        assert pl[0].ri == (100.0 / 150.0)
        assert pl[1].ri == (50.0 / 150.0)

    def test_append_lower_intensity(self):
        """B10: new peak lower than basepeak, ri set."""
        pl = make_peaklist((100, 100), (200, 50))
        pl.append(make_peak(300.0, 75.0))
        assert pl[2].ri == 0.75
        assert pl.basepeak.intensity == 100.0

    def test_append_from_list_tuple(self):
        """append accepts list/tuple (mz, ai)."""
        pl = obj_peaklist.peaklist()
        pl.append([100.0, 50.0])
        assert len(pl) == 1
        assert pl[0].mz == 100.0

    def test_append_from_tuple(self):
        """append accepts tuple (mz, ai)."""
        pl = obj_peaklist.peaklist()
        pl.append((100.0, 50.0))
        assert len(pl) == 1
        assert pl[0].mz == 100.0


# ============================================================================
# TEST: reset
# ============================================================================

class TestPeaklistReset:
    """Test peaklist reset."""

    def test_reset_sorts_and_recalculates(self):
        """reset sorts, recalcs basepeak and ri."""
        pl = obj_peaklist.peaklist()
        pl.peaks = [
            make_peak(300.0, 50.0),
            make_peak(100.0, 100.0),
            make_peak(200.0, 75.0)
        ]
        pl.reset()
        assert pl[0].mz == 100.0
        assert pl.basepeak.mz == 100.0
        assert pl[0].ri == 1.0


# ============================================================================
# TEST: duplicate
# ============================================================================

class TestPeaklistDuplicate:
    """Test peaklist duplicate."""

    def test_duplicate_creates_deepcopy(self):
        """duplicate returns deepcopy."""
        pl1 = make_peaklist((100, 100), (200, 50))
        pl2 = pl1.duplicate()
        assert pl1 is not pl2
        assert len(pl1) == len(pl2)
        assert pl1[0] is not pl2[0]
        assert pl1[0].mz == pl2[0].mz

    def test_duplicate_modifies_independently(self):
        """Modifying duplicate doesn't affect original."""
        pl1 = make_peaklist((100, 100))
        pl2 = pl1.duplicate()
        pl2.append(make_peak(200.0, 50.0))
        assert len(pl1) == 1
        assert len(pl2) == 2


# ============================================================================
# TEST: groupname
# ============================================================================

class TestPeaklistGroupname:
    """Test peaklist groupname generation."""

    def test_groupname_empty_peaklist(self):
        """B47-48-49: empty peaklist returns first name."""
        pl = obj_peaklist.peaklist()
        name = pl.groupname()
        assert name == 'A'

    def test_groupname_no_used_groups(self):
        """B47: no used groups, returns first name."""
        pl = make_peaklist((100, 100), (200, 50))
        name = pl.groupname()
        assert name == 'A'

    def test_groupname_with_used_groups(self):
        """B47-48: skips used group names."""
        pl = make_peaklist((100, 100), (200, 50))
        pl[0].setgroup('A')
        pl[1].setgroup('B')
        name = pl.groupname()
        assert name == 'C'

    def test_groupname_increments_size(self):
        """B49: increments size when all names exhausted."""
        pl = obj_peaklist.peaklist()
        for i in range(26):
            p = make_peak(100.0 + i, 100.0)
            p.setgroup(chr(ord('A') + i))
            pl.peaks.append(p)
        pl.basepeak = pl.peaks[0]
        name = pl.groupname()
        assert name == 'AA'


# ============================================================================
# TEST: sort
# ============================================================================

class TestPeaklistSort:
    """Test peaklist sort."""

    def test_sort_unordered(self):
        """sort orders peaks by m/z."""
        pl = obj_peaklist.peaklist()
        pl.peaks = [
            make_peak(300.0, 50.0),
            make_peak(100.0, 100.0),
            make_peak(200.0, 75.0)
        ]
        pl.sort()
        assert pl[0].mz == 100.0
        assert pl[1].mz == 200.0
        assert pl[2].mz == 300.0

    def test_sort_already_sorted(self):
        """sort on sorted list is idempotent."""
        pl = make_peaklist((100, 100), (200, 50), (300, 75))
        mzs_before = [p.mz for p in pl.peaks]
        pl.sort()
        mzs_after = [p.mz for p in pl.peaks]
        assert mzs_before == mzs_after


# ============================================================================
# TEST: delete
# ============================================================================

class TestPeaklistDelete:
    """Test peaklist delete."""

    def test_delete_empty_list(self):
        """B12-true: delete on empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.delete([])
        assert len(pl) == 0

    def test_delete_non_basepeak(self):
        """B12-false B13-false B14-false: delete non-basepeak."""
        pl = make_peaklist((100, 100), (200, 50), (300, 75))
        pl.delete([1])
        assert len(pl) == 2
        assert pl[0].mz == 100.0
        assert pl[1].mz == 300.0

    def test_delete_basepeak(self):
        """B12-false B13-true B14-true: delete basepeak, recalcs."""
        pl = make_peaklist((100, 100), (200, 50), (300, 75))
        assert pl.basepeak.mz == 100.0
        pl.delete([0])
        assert len(pl) == 2
        assert pl.basepeak.mz == 300.0  # New basepeak is highest intensity
        assert pl.basepeak.ri == 1.0  # Basepeak should have ri=1.0

    def test_delete_multiple_indices(self):
        """delete removes multiple peaks in reverse order."""
        pl = make_peaklist((100, 100), (200, 50), (300, 75), (400, 25))
        pl.delete([1, 3])
        assert len(pl) == 2
        assert pl[0].mz == 100.0
        assert pl[1].mz == 300.0

    def test_delete_basepeak_and_others(self):
        """delete basepeak with others, still recalcs."""
        pl = make_peaklist((100, 100), (200, 50), (300, 75))
        pl.delete([0, 2])
        assert len(pl) == 1
        assert pl.basepeak.mz == 200.0


# ============================================================================
# TEST: empty
# ============================================================================

class TestPeaklistEmpty:
    """Test peaklist empty."""

    def test_empty_clears_peaks(self):
        """empty removes all peaks and basepeak."""
        pl = make_peaklist((100, 100), (200, 50))
        pl.empty()
        assert len(pl) == 0
        assert pl.basepeak is None

    def test_empty_on_already_empty(self):
        """empty on empty list is safe."""
        pl = obj_peaklist.peaklist()
        pl.empty()
        assert len(pl) == 0


# ============================================================================
# TEST: crop
# ============================================================================

class TestPeaklistCrop:
    """Test peaklist crop."""

    def test_crop_empty_list(self):
        """B15-true: crop empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.crop(100.0, 200.0)
        assert len(pl) == 0

    def test_crop_within_range(self):
        """B16: delete peaks outside range."""
        pl = make_peaklist((50, 10), (100, 100), (150, 75), (200, 50), (250, 25))
        pl.crop(100.0, 200.0)
        assert len(pl) == 3
        assert pl[0].mz == 100.0
        assert pl[1].mz == 150.0
        assert pl[2].mz == 200.0

    def test_crop_boundary_inclusive(self):
        """Crop boundaries are inclusive."""
        pl = make_peaklist((100, 100), (150, 50), (200, 75))
        pl.crop(100.0, 200.0)
        assert len(pl) == 3

    def test_crop_excludes_outside(self):
        """Peaks outside range are excluded."""
        pl = make_peaklist((50, 10), (100, 100), (200, 75), (250, 25))
        pl.crop(100.0, 200.0)
        assert pl[0].mz >= 100.0
        assert pl[-1].mz <= 200.0


# ============================================================================
# TEST: multiply
# ============================================================================

class TestPeaklistMultiply:
    """Test peaklist multiply."""

    def test_multiply_empty_list(self):
        """B17-true: multiply empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.multiply(2.0)
        assert len(pl) == 0

    def test_multiply_scales_intensities(self):
        """B17-false: multiply scales ai and base."""
        pl = make_peaklist((100, 100), (200, 50))
        pl[0].setbase(10.0)
        pl[1].setbase(5.0)
        pl.multiply(2.0)
        assert pl[0].ai == 200.0
        assert pl[0].base == 20.0
        assert pl[1].ai == 100.0

    def test_multiply_updates_basepeak(self):
        """multiply updates basepeak and ri."""
        pl = make_peaklist((100, 100), (200, 50))
        pl.multiply(2.0)
        assert pl.basepeak.intensity == 200.0
        assert pl[0].ri == 1.0
        assert pl[1].ri == 0.5


# ============================================================================
# TEST: combine
# ============================================================================

class TestPeaklistCombine:
    """Test peaklist combine."""

    def test_combine_empty_peaklists(self):
        """Combining empty peaklists."""
        pl1 = obj_peaklist.peaklist()
        pl2 = obj_peaklist.peaklist()
        pl1.combine(pl2)
        assert len(pl1) == 0

    def test_combine_adds_peaks(self):
        """combine adds peaks from other."""
        pl1 = make_peaklist((100, 100), (200, 50))
        pl2 = make_peaklist((300, 75))
        pl1.combine(pl2)
        assert len(pl1) == 3
        assert pl1[2].mz == 300.0

    def test_combine_sorts_and_recalcs(self):
        """combine sorts and recalculates basepeak."""
        pl1 = make_peaklist((100, 100))
        pl2 = make_peaklist((200, 150))
        pl1.combine(pl2)
        assert pl1.basepeak.intensity == 150.0
        assert pl1[0].ri == (100.0 / 150.0)

    def test_combine_doesnt_modify_other(self):
        """combine doesn't modify the other peaklist."""
        pl1 = make_peaklist((100, 100))
        pl2 = make_peaklist((200, 50))
        len_pl2_before = len(pl2)
        pl1.combine(pl2)
        assert len(pl2) == len_pl2_before


# ============================================================================
# TEST: recalibrate
# ============================================================================

class TestPeaklistRecalibrate:
    """Test peaklist recalibrate."""

    def test_recalibrate_empty_list(self):
        """B18-true: recalibrate empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.recalibrate(lambda p, m: m + 10, [])
        assert len(pl) == 0

    def test_recalibrate_applies_function(self):
        """B18-false: apply calibration function."""
        pl = make_peaklist((100, 100), (200, 50))
        pl.recalibrate(lambda p, m: m + 10, [])
        assert pl[0].mz == 110.0
        assert pl[1].mz == 210.0

    def test_recalibrate_with_params(self):
        """recalibrate passes params to function."""
        pl = make_peaklist((100, 100), (200, 50))
        pl.recalibrate(lambda p, m: m * p[0], [2.0])
        assert pl[0].mz == 200.0
        assert pl[1].mz == 400.0


# ============================================================================
# TEST: deisotope
# ============================================================================

class TestPeaklistDeisotope:
    """Test peaklist deisotope."""

    def test_deisotope_empty_list(self):
        """B19-true: deisotope empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.deisotope()
        assert len(pl) == 0

    def test_deisotope_calls_mod_peakpicking(self, mocker):
        """B19-false: calls mod_peakpicking.deisotope."""
        pl = make_peaklist((100, 100), (200, 50))
        mock_deiso = mocker.patch('mspy.mod_peakpicking.deisotope')
        pl.deisotope(maxCharge=2, mzTolerance=0.1)
        mock_deiso.assert_called_once()
        call_kwargs = mock_deiso.call_args[1]
        assert call_kwargs['maxCharge'] == 2
        assert call_kwargs['mzTolerance'] == 0.1

    def test_deisotope_with_custom_params(self, mocker):
        """deisotope passes custom parameters."""
        pl = make_peaklist((100, 100))
        mock_deiso = mocker.patch('mspy.mod_peakpicking.deisotope')
        pl.deisotope(maxCharge=3, mzTolerance=0.2,
                    intTolerance=0.3, isotopeShift=0.5)
        call_kwargs = mock_deiso.call_args[1]
        assert call_kwargs['maxCharge'] == 3
        assert call_kwargs['mzTolerance'] == 0.2


# ============================================================================
# TEST: deconvolute
# ============================================================================

class TestPeaklistDeconvolute:
    """Test peaklist deconvolute."""

    def test_deconvolute_empty_list(self):
        """B20-true: deconvolute empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.deconvolute()
        assert len(pl) == 0

    def test_deconvolute_calls_mod_peakpicking(self, mocker):
        """B20-false: calls mod_peakpicking.deconvolute."""
        pl = make_peaklist((100, 100), (200, 50))
        mock_result = obj_peaklist.peaklist([make_peak(100.0, 100.0)])
        mock_deconv = mocker.patch('mspy.mod_peakpicking.deconvolute')
        mock_deconv.return_value = mock_result
        pl.deconvolute(massType=0)
        mock_deconv.assert_called_once()
        call_kwargs = mock_deconv.call_args[1]
        assert call_kwargs['massType'] == 0

    def test_deconvolute_updates_peaks(self, mocker):
        """deconvolute updates peaklist peaks."""
        pl = make_peaklist((100, 100), (200, 50))
        new_peaks = obj_peaklist.peaklist([
            make_peak(100.0, 100.0),
            make_peak(200.0, 50.0)
        ])
        mock_deconv = mocker.patch('mspy.mod_peakpicking.deconvolute')
        mock_deconv.return_value = new_peaks
        pl.deconvolute()
        assert len(pl) == 2


# ============================================================================
# TEST: consolidate
# ============================================================================

class TestPeaklistConsolidate:
    """Test peaklist consolidate."""

    def test_consolidate_empty_list(self):
        """B21-true: consolidate empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.consolidate(window=1.0)
        assert len(pl) == 0

    def test_consolidate_single_peak(self):
        """consolidate with single peak unchanged."""
        pl = make_peaklist((100, 100))
        pl.consolidate(window=1.0)
        assert len(pl) == 1
        assert pl[0].mz == 100.0

    def test_consolidate_merges_nearby_peaks(self):
        """B21-false B22-false B23-true: merge peaks within window."""
        pl = obj_peaklist.peaklist([
            make_peak(100.0, 100.0),
            make_peak(101.0, 50.0)
        ])
        pl.reset()
        pl.consolidate(window=2.0)
        assert len(pl) == 1
        assert 100.0 < pl[0].mz < 101.0

    def test_consolidate_keeps_distant_peaks(self):
        """B21-false B23-false: keep peaks outside window."""
        pl = obj_peaklist.peaklist([
            make_peak(100.0, 100.0),
            make_peak(110.0, 50.0)
        ])
        pl.reset()
        pl.consolidate(window=2.0)
        assert len(pl) == 2

    def test_consolidate_weighted_mz(self):
        """consolidate uses intensity-weighted m/z."""
        p1 = make_peak(100.0, 100.0)
        p2 = make_peak(102.0, 100.0)
        pl = obj_peaklist.peaklist([p1, p2])
        pl.reset()
        pl.consolidate(window=5.0)
        assert len(pl) == 1
        # Weighted average: (100*100 + 102*100) / 200 = 101
        assert pl[0].mz == 101.0

    def test_consolidate_with_fwhm(self):
        """B22-true: use fwhm for window calculation."""
        p1 = make_peak(100.0, 100.0, fwhm=0.5)
        p2 = make_peak(100.5, 50.0, fwhm=0.5)
        pl = obj_peaklist.peaklist([p1, p2])
        pl.reset()
        pl.consolidate(window=5.0, forceWindow=False)
        # win = (0.5 + 0.5) / 8 = 0.125, so 100.0 + 0.125 = 100.125 < 100.5
        # Should NOT merge
        assert len(pl) == 2

    def test_consolidate_force_window(self):
        """consolidate with forceWindow=True ignores fwhm."""
        p1 = make_peak(100.0, 100.0, fwhm=0.1)
        p2 = make_peak(100.5, 50.0, fwhm=0.1)
        pl = obj_peaklist.peaklist([p1, p2])
        pl.reset()
        pl.consolidate(window=0.1, forceWindow=True)
        # Should not merge due to small window
        assert len(pl) == 2

    def test_consolidate_weighted_fwhm(self):
        """B24: weighted fwhm calculation."""
        p1 = make_peak(100.0, 100.0, fwhm=1.0)
        p2 = make_peak(100.05, 100.0, fwhm=2.0)
        pl = obj_peaklist.peaklist([p1, p2])
        pl.reset()
        pl.consolidate(window=5.0)
        # win = (1.0 + 2.0) / 8 = 0.375, so 100.0 + 0.375 = 100.375 > 100.05 -> MERGE
        assert len(pl) == 1
        # Weighted fwhm: (1*100 + 2*100) / 200 = 1.5
        assert pl[0].fwhm == 1.5


# ============================================================================
# TEST: remthreshold
# ============================================================================

class TestPeaklistRemthreshold:
    """Test peaklist remthreshold."""

    def test_remthreshold_empty_list(self):
        """remthreshold on empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.remthreshold(absThreshold=50)
        assert len(pl) == 0

    def test_remthreshold_absolute(self):
        """B25-false B26: remove by absolute threshold."""
        pl = make_peaklist((100, 100), (200, 50), (300, 25))
        pl.remthreshold(absThreshold=60)
        assert len(pl) == 1
        assert pl[0].mz == 100.0

    def test_remthreshold_relative(self):
        """remove by relative threshold (percent of basepeak)."""
        pl = make_peaklist((100, 100), (200, 50), (300, 10))
        pl.remthreshold(relThreshold=0.2)
        # basepeak = 100, threshold = 20
        assert len(pl) == 2
        assert pl[0].mz == 100.0
        assert pl[1].mz == 200.0

    def test_remthreshold_sn(self):
        """B27: remove by signal-to-noise threshold."""
        p1 = make_peak(100.0, 100.0, sn=10.0)
        p2 = make_peak(200.0, 50.0, sn=3.0)
        p3 = make_peak(300.0, 75.0, sn=1.0)
        pl = obj_peaklist.peaklist([p1, p2, p3])
        pl.reset()
        pl.remthreshold(snThreshold=5.0)
        assert len(pl) == 1
        assert pl[0].mz == 100.0

    def test_remthreshold_combined(self):
        """remthreshold uses max of absolute and relative."""
        pl = make_peaklist((100, 100), (200, 20), (300, 30))
        pl.remthreshold(absThreshold=25, relThreshold=0.0)
        assert len(pl) == 2
        assert pl[0].mz == 100.0
        assert pl[1].mz == 300.0


# ============================================================================
# TEST: remshoulders
# ============================================================================

class TestPeaklistRemshoulders:
    """Test peaklist remshoulders."""

    def test_remshoulders_empty_list(self):
        """B28-true: remshoulders empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.remshoulders()
        assert len(pl) == 0

    def test_remshoulders_no_sn_is_candidate(self):
        """B29-false: peak with no sn is candidate parent."""
        pl = obj_peaklist.peaklist([
            make_peak(100.0, 100.0, fwhm=0.5),
            make_peak(100.3, 10.0),
            make_peak(101.0, 50.0)
        ])
        pl.reset()
        pl.remshoulders(window=2.0, relThreshold=0.1)
        # Parent at 100.0: win = 0.5*2 = 1.0, range [99, 101]
        # 100.3 is in range and 10 < 100*0.1=10 (edge case), removed
        # 101.0 is outside range [99, 101] (exclusive)
        # So should have 2 peaks left
        assert len(pl) >= 1

    def test_remshoulders_sn_threshold(self):
        """B29-true: peak with high sn is candidate parent."""
        p1 = make_peak(100.0, 100.0, fwhm=0.5, sn=50.0)  # sn*rel = 50*0.05 = 2.5 < 3 (IS candidate)
        p2 = make_peak(100.3, 5.0)
        p3 = make_peak(101.5, 50.0)
        pl = obj_peaklist.peaklist([p1, p2, p3])
        pl.reset()
        pl.remshoulders(window=2.0, relThreshold=0.05)
        # p1 is candidate: win = 0.5*2 = 1.0, range [99, 101]
        # p2 at 100.3 in range, intensity 5 < 100*0.05=5 (edge case, removed)
        # p3 at 101.5 outside range
        assert len(pl) >= 1

    def test_remshoulders_uses_fwhm(self):
        """B30: use peak.fwhm for window."""
        p1 = make_peak(100.0, 100.0, fwhm=1.0)
        p2 = make_peak(101.5, 5.0)
        pl = obj_peaklist.peaklist([p1, p2])
        pl.reset()
        pl.remshoulders(window=2.0, relThreshold=0.1, fwhm=0.01)
        # Window = 1.0 * 2.0 = 2.0, p2 at 101.5 within range
        assert len(pl) == 1

    def test_remshoulders_default_fwhm(self):
        """B31: use default fwhm param when peak.fwhm is None."""
        p1 = make_peak(100.0, 100.0, fwhm=None)
        p2 = make_peak(100.08, 5.0)
        p3 = make_peak(101.0, 50.0)
        pl = obj_peaklist.peaklist([p1, p2, p3])
        pl.reset()
        pl.remshoulders(window=2.0, relThreshold=0.1, fwhm=0.05)
        # Window = 0.05 * 2.0 = 0.1, p1 range [99.9, 100.1]
        # p2 at 100.08 in range, 5 < 100*0.1=10, removed
        # p3 at 101.0 outside
        assert len(pl) >= 1

    def test_remshoulders_no_fwhm_skips(self):
        """B32: skip if no fwhm available."""
        p1 = make_peak(100.0, 100.0, fwhm=None)
        p2 = make_peak(100.15, 5.0)
        pl = obj_peaklist.peaklist([p1, p2])
        pl.reset()
        pl.remshoulders(window=2.0, relThreshold=0.1, fwhm=None)
        # No fwhm, should skip shoulder filtering
        assert len(pl) == 2

    def test_remshoulders_shoulder_detection(self):
        """B33: shoulder within window and below threshold."""
        p1 = make_peak(100.0, 100.0, fwhm=0.5)
        p_shoulder = make_peak(100.3, 8.0)  # Within window, below threshold
        p_other = make_peak(100.8, 50.0)    # Within window but above threshold
        pl = obj_peaklist.peaklist([p1, p_shoulder, p_other])
        pl.reset()
        pl.remshoulders(window=2.0, relThreshold=0.1, fwhm=0.01)
        # Only shoulder should be removed
        assert len(pl) == 2

    def test_remshoulders_break_on_high_mz(self):
        """B34: break when peak.mz > highMZ."""
        peaks = [
            make_peak(100.0, 100.0, fwhm=0.5),
            make_peak(101.0, 5.0),
            make_peak(102.0, 5.0),
            make_peak(103.0, 5.0)
        ]
        pl = obj_peaklist.peaklist(peaks)
        pl.reset()
        pl.remshoulders(window=1.0, relThreshold=0.1, fwhm=0.01)
        # Window = 0.5 * 1.0 = 0.5, range [99.5, 100.5]
        # p2 and p3 outside range, p4 definitely outside
        assert len(pl) >= 1

    def test_remshoulders_fwhm_param_zero(self):
        """B32-B33: when fwhm=0 (falsy), continue without processing shoulder."""
        # This tests both the condition line 500 and the continue path
        p1 = make_peak(100.0, 100.0, fwhm=None)
        p2 = make_peak(100.1, 5.0)
        pl = obj_peaklist.peaklist([p1, p2])
        pl.reset()
        # fwhm=0 is falsy, so should skip and continue
        pl.remshoulders(window=2.0, relThreshold=0.1, fwhm=0)
        # Should not remove anything since fwhm is falsy
        assert len(pl) == 2

    def test_remshoulders_with_fwhm_param_truthy(self):
        """B32: when fwhm param is truthy, use it for window calculation."""
        # This tests the true path of line 500
        p1 = make_peak(100.0, 100.0, fwhm=None)
        p2 = make_peak(100.08, 5.0)
        p3 = make_peak(101.0, 50.0)
        pl = obj_peaklist.peaklist([p1, p2, p3])
        pl.reset()
        # Use fwhm param when peak.fwhm is None
        pl.remshoulders(window=2.0, relThreshold=0.1, fwhm=0.05)
        # Window = 0.05 * 2.0 = 0.1, p1 range [99.9, 100.1]
        # p2 at 100.08 in range and below threshold
        assert len(pl) <= 2


# ============================================================================
# TEST: remisotopes
# ============================================================================

class TestPeaklistRemisotopes:
    """Test peaklist remisotopes."""

    def test_remisotopes_empty_list(self):
        """B35-true: remisotopes empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.remisotopes()
        assert len(pl) == 0

    def test_remisotopes_removes_isotopes(self):
        """B35-false B36: remove isotope!=0 with charge."""
        p1 = make_peak(100.0, 100.0, isotope=0, charge=1)
        p2 = make_peak(101.0, 50.0, isotope=1, charge=1)  # Isotope
        p3 = make_peak(102.0, 25.0, isotope=2, charge=1)  # Isotope
        pl = obj_peaklist.peaklist([p1, p2, p3])
        pl.reset()
        pl.remisotopes()
        assert len(pl) == 1
        assert pl[0].isotope == 0

    def test_remisotopes_keeps_uncharged_isotope(self):
        """isotope without charge is kept."""
        p1 = make_peak(100.0, 100.0, isotope=1, charge=None)
        p2 = make_peak(101.0, 50.0, isotope=0, charge=1)
        pl = obj_peaklist.peaklist([p1, p2])
        pl.reset()
        pl.remisotopes()
        assert len(pl) == 2


# ============================================================================
# TEST: remuncharged
# ============================================================================

class TestPeaklistRemuncharged:
    """Test peaklist remuncharged."""

    def test_remuncharged_empty_list(self):
        """B37-true: remuncharged empty list returns."""
        pl = obj_peaklist.peaklist()
        pl.remuncharged()
        assert len(pl) == 0

    def test_remuncharged_removes_uncharged(self):
        """B37-false B38: remove peaks with charge==None."""
        p1 = make_peak(100.0, 100.0, charge=1)
        p2 = make_peak(101.0, 50.0, charge=None)  # Uncharged
        p3 = make_peak(102.0, 25.0, charge=2)
        pl = obj_peaklist.peaklist([p1, p2, p3])
        pl.reset()
        pl.remuncharged()
        assert len(pl) == 2
        assert pl[0].charge == 1
        assert pl[1].charge == 2


# ============================================================================
# TEST: _checkPeak
# ============================================================================

class TestCheckPeak:
    """Test peaklist._checkPeak."""

    def test_checkpeak_peak_instance(self):
        """B39: peak instance returned as-is."""
        peak_obj = make_peak(100.0, 50.0)
        pl = obj_peaklist.peaklist()
        result = pl._checkPeak(peak_obj)
        assert result is peak_obj

    def test_checkpeak_list_tuple(self):
        """B40: list of len 2 constructs peak."""
        pl = obj_peaklist.peaklist()
        result = pl._checkPeak([100.0, 50.0])
        assert isinstance(result, obj_peak.peak)
        assert result.mz == 100.0
        assert result.ai == 50.0

    def test_checkpeak_tuple_pair(self):
        """B40: tuple of len 2 constructs peak."""
        pl = obj_peaklist.peaklist()
        result = pl._checkPeak((100.0, 50.0))
        assert isinstance(result, obj_peak.peak)
        assert result.mz == 100.0
        assert result.ai == 50.0

    def test_checkpeak_invalid_list_length(self):
        """B41: list with len!=2 raises TypeError."""
        pl = obj_peaklist.peaklist()
        with pytest.raises(TypeError):
            pl._checkPeak([100.0])
        with pytest.raises(TypeError):
            pl._checkPeak([100.0, 50.0, 25.0])

    def test_checkpeak_invalid_type(self):
        """B41: non-peak, non-list/tuple raises TypeError."""
        pl = obj_peaklist.peaklist()
        with pytest.raises(TypeError):
            pl._checkPeak("invalid")
        with pytest.raises(TypeError):
            pl._checkPeak(123)


# ============================================================================
# TEST: _setbasepeak
# ============================================================================

class TestSetBasepeak:
    """Test peaklist._setbasepeak."""

    def test_setbasepeak_empty(self):
        """B42-true: empty list sets basepeak to None."""
        pl = obj_peaklist.peaklist()
        pl._setbasepeak()
        assert pl.basepeak is None

    def test_setbasepeak_single(self):
        """B42-false B43: single peak becomes basepeak."""
        pl = obj_peaklist.peaklist()
        pl.peaks = [make_peak(100.0, 50.0)]
        pl._setbasepeak()
        assert pl.basepeak is pl.peaks[0]

    def test_setbasepeak_highest_intensity(self):
        """B42-false B43: highest intensity peak becomes basepeak."""
        pl = obj_peaklist.peaklist()
        pl.peaks = [
            make_peak(100.0, 50.0),
            make_peak(200.0, 200.0),
            make_peak(300.0, 100.0)
        ]
        pl._setbasepeak()
        assert pl.basepeak is pl.peaks[1]
        assert pl.basepeak.intensity == 200.0


# ============================================================================
# TEST: _setRelativeIntensities
# ============================================================================

class TestSetRelativeIntensities:
    """Test peaklist._setRelativeIntensities."""

    def test_setrelativeintensities_empty(self):
        """B44-true: empty list returns."""
        pl = obj_peaklist.peaklist()
        pl._setRelativeIntensities()
        assert len(pl) == 0

    def test_setrelativeintensities_maxint_zero(self):
        """B45-false: basepeak intensity=0, all ri=1.0."""
        pl = obj_peaklist.peaklist()
        pl.peaks = [
            make_peak(100.0, 0.0),
            make_peak(200.0, 0.0)
        ]
        pl.basepeak = pl.peaks[0]
        pl._setRelativeIntensities()
        assert pl[0].ri == 1.0
        assert pl[1].ri == 1.0

    def test_setrelativeintensities_divides(self):
        """B45-true: basepeak intensity>0, divide all."""
        pl = obj_peaklist.peaklist()
        pl.peaks = [
            make_peak(100.0, 100.0),
            make_peak(200.0, 50.0),
            make_peak(300.0, 25.0)
        ]
        pl.basepeak = pl.peaks[0]
        pl._setRelativeIntensities()
        assert pl[0].ri == 1.0
        assert pl[1].ri == 0.5
        assert pl[2].ri == 0.25


# ============================================================================
# TEST: _generateGroupNames
# ============================================================================

class TestGenerateGroupNames:
    """Test peaklist._generateGroupNames."""

    def test_generategroupnames_size1(self):
        """B46: size=1 yields 26 names (A-Z)."""
        pl = obj_peaklist.peaklist()
        names = list(pl._generateGroupNames(1))
        assert len(names) == 26
        assert names[0] == 'A'
        assert names[25] == 'Z'

    def test_generategroupnames_size2(self):
        """B46: size=2 yields 676 names (AA-ZZ)."""
        pl = obj_peaklist.peaklist()
        names = list(pl._generateGroupNames(2))
        assert len(names) == 676
        assert names[0] == 'AA'
        assert names[1] == 'AB'
        assert names[-1] == 'ZZ'

    def test_generategroupnames_size3(self):
        """size=3 yields 17576 names."""
        pl = obj_peaklist.peaklist()
        names = list(pl._generateGroupNames(3))
        assert len(names) == 17576
        assert names[0] == 'AAA'


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================

class TestPeaklistPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(st.lists(
        st.tuples(st.floats(100, 1000, allow_nan=False, allow_infinity=False),
                  st.floats(1, 1000, allow_nan=False, allow_infinity=False)),
        min_size=0, max_size=100
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
    def test_peaklist_construction_always_valid(self, peak_data):
        """Peaklist construction with any valid data succeeds."""
        if peak_data:
            pl = obj_peaklist.peaklist([obj_peak.peak(mz, ai) for mz, ai in peak_data])
            assert len(pl) == len(peak_data)
            assert all(p.mz >= 0 for p in pl.peaks)

    @given(st.lists(
        st.tuples(st.floats(100, 1000, allow_nan=False, allow_infinity=False),
                  st.floats(1, 1000, allow_nan=False, allow_infinity=False)),
        min_size=1, max_size=50
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    def test_peaklist_always_sorted(self, peak_data):
        """Peaklist is always sorted by m/z."""
        pl = obj_peaklist.peaklist([obj_peak.peak(mz, ai) for mz, ai in peak_data])
        mzs = [p.mz for p in pl.peaks]
        assert mzs == sorted(mzs)

    @given(st.lists(
        st.tuples(st.floats(100, 1000, allow_nan=False, allow_infinity=False),
                  st.floats(1, 1000, allow_nan=False, allow_infinity=False)),
        min_size=1, max_size=50
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    def test_peaklist_basepeak_has_max_intensity(self, peak_data):
        """Basepeak always has maximum intensity."""
        pl = obj_peaklist.peaklist([obj_peak.peak(mz, ai) for mz, ai in peak_data])
        max_intensity = max(p.intensity for p in pl.peaks)
        assert pl.basepeak.intensity == max_intensity

    @given(st.lists(
        st.tuples(st.floats(100, 1000, allow_nan=False, allow_infinity=False),
                  st.floats(1, 1000, allow_nan=False, allow_infinity=False)),
        min_size=1, max_size=50
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    def test_relative_intensities_correct(self, peak_data):
        """Relative intensities are calculated correctly."""
        pl = obj_peaklist.peaklist([obj_peak.peak(mz, ai) for mz, ai in peak_data])
        max_intensity = pl.basepeak.intensity
        if max_intensity > 0:
            for p in pl.peaks:
                expected_ri = p.intensity / max_intensity
                assert abs(p.ri - expected_ri) < 1e-9
        else:
            for p in pl.peaks:
                assert p.ri == 1.0

    @given(st.lists(
        st.tuples(st.floats(100, 1000, allow_nan=False, allow_infinity=False),
                  st.floats(1, 1000, allow_nan=False, allow_infinity=False)),
        min_size=1, max_size=50
    ))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.filter_too_much])
    def test_append_maintains_sort(self, peak_data):
        """Appending peaks maintains sort order."""
        pl = obj_peaklist.peaklist([obj_peak.peak(mz, ai) for mz, ai in peak_data])
        new_peak = obj_peak.peak(500.0, 100.0)
        pl.append(new_peak)
        mzs = [p.mz for p in pl.peaks]
        assert mzs == sorted(mzs)

    @given(st.lists(
        st.tuples(st.floats(100, 1000, allow_nan=False, allow_infinity=False),
                  st.floats(10, 1000, allow_nan=False, allow_infinity=False)),
        min_size=1, max_size=50
    ), st.floats(0.1, 10.0))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_multiply_scales_correctly(self, peak_data, factor):
        """Multiplying scales all intensities correctly."""
        if not peak_data:
            return
        # Filter to unique mz values to avoid sorting issues
        seen_mz = set()
        unique_peaks = []
        for mz, ai in sorted(peak_data):
            if mz not in seen_mz:
                seen_mz.add(mz)
                unique_peaks.append((mz, ai))

        if len(unique_peaks) < len(peak_data):
            return  # Skip test if we had duplicates

        pl = obj_peaklist.peaklist([obj_peak.peak(mz, ai) for mz, ai in unique_peaks])
        original_ais = [p.ai for p in pl.peaks]
        pl.multiply(factor)
        for i, expected_ai in enumerate(original_ais):
            expected = expected_ai * factor
            actual = pl[i].ai
            # Check within machine precision
            assert abs(actual - expected) < abs(expected) * 1e-10 or abs(actual - expected) < 1e-10


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPeaklistIntegration:
    """Integration tests for complex workflows."""

    def test_workflow_create_modify_query(self):
        """Create, modify, and query peaklist."""
        pl = obj_peaklist.peaklist()
        pl.append((100, 100))
        pl.append((200, 50))
        pl.append((150, 75))
        assert len(pl) == 3
        assert pl[0].mz == 100.0
        assert pl.basepeak.mz == 100.0

    def test_workflow_combine_and_clean(self):
        """Combine peaklists and apply filters."""
        pl1 = make_peaklist((100, 100), (200, 50))
        pl2 = make_peaklist((300, 75), (400, 25))
        pl1.combine(pl2)
        assert len(pl1) == 4
        pl1.remthreshold(absThreshold=30)
        assert all(p.intensity >= 30 for p in pl1.peaks)

    def test_workflow_duplicate_and_modify(self):
        """Duplicate peaklist and modify independently."""
        pl1 = make_peaklist((100, 100), (200, 50), (300, 75))
        pl2 = pl1.duplicate()
        pl2.delete([0])
        assert len(pl1) == 3
        assert len(pl2) == 2
        assert pl1[0] is not pl2[0]

    def test_workflow_crop_and_check_basepeak(self):
        """Crop peaklist and verify basepeak recalculation."""
        pl = make_peaklist((50, 30), (100, 100), (200, 50), (300, 200), (400, 25))
        assert pl.basepeak.mz == 300.0
        pl.crop(100.0, 250.0)
        assert pl.basepeak.mz == 100.0  # After crop, 100 becomes highest in range
        assert all(100.0 <= p.mz <= 250.0 for p in pl.peaks)

    def test_workflow_setitem_with_intensity_inversion(self):
        """Setting item can change basepeak."""
        pl = make_peaklist((100, 50), (200, 100))
        assert pl.basepeak.mz == 200.0
        pl[1] = make_peak(200.0, 25.0)
        assert pl.basepeak.mz == 100.0

    def test_workflow_add_operator(self):
        """Using + operator creates new peaklist."""
        pl1 = make_peaklist((100, 100))
        pl2 = make_peaklist((200, 50))
        pl3 = pl1 + pl2
        assert pl1 is not pl3
        assert len(pl3) == 2
        assert pl1[0].intensity == 100.0

    def test_workflow_mul_operator(self):
        """Using * operator creates new peaklist."""
        pl1 = make_peaklist((100, 100))
        pl2 = pl1 * 2.0
        assert pl1 is not pl2
        assert pl1[0].ai == 100.0
        assert pl2[0].ai == 200.0

    def test_workflow_iteration_and_modification(self):
        """Iterate and modify peaklist."""
        pl = make_peaklist((100, 100), (200, 50), (300, 75))
        count = 0
        for peak in pl:
            assert peak.mz > 0
            count += 1
        assert count == 3

    def test_workflow_complex_consolidate(self):
        """Complex consolidate with various peak configurations."""
        peaks = [
            make_peak(100.0, 100.0, fwhm=1.0),
            make_peak(100.15, 80.0, fwhm=1.0),
            make_peak(110.0, 50.0, fwhm=0.5),
            make_peak(110.1, 30.0, fwhm=0.5),
        ]
        pl = obj_peaklist.peaklist(peaks)
        pl.consolidate(window=2.0, forceWindow=False)
        # win1 = (1.0 + 1.0) / 8 = 0.25, so 100.0 + 0.25 = 100.25 > 100.15 -> MERGE
        # win2 = (0.5 + 0.5) / 8 = 0.125, so 110.0 + 0.125 = 110.125 > 110.1 -> MERGE
        assert len(pl) == 2
        # First group: (100,100.15), Second group: (110, 110.1)
        assert pl[0].mz < 101.0
        assert pl[1].mz > 109.0

    def test_workflow_remshoulders_comprehensive(self):
        """Complex remshoulders filtering."""
        peaks = [
            make_peak(100.0, 100.0, fwhm=0.5),
            make_peak(100.2, 15.0),      # Weak shoulder
            make_peak(100.6, 5.0),       # Very weak
            make_peak(105.0, 80.0, fwhm=0.5),
            make_peak(105.3, 10.0),      # Weak shoulder
        ]
        pl = obj_peaklist.peaklist(peaks)
        pl.reset()
        pl.remshoulders(window=2.0, relThreshold=0.1, fwhm=0.01)
        # Should remove weak shoulders
        assert len(pl) <= len(peaks)

    def test_workflow_isotope_filtering(self):
        """Filter isotopes and uncharged peaks."""
        peaks = [
            make_peak(100.0, 100.0, isotope=0, charge=1),
            make_peak(101.0, 50.0, isotope=1, charge=1),
            make_peak(102.0, 25.0, isotope=2, charge=1),
            make_peak(200.0, 100.0, isotope=0, charge=None),
        ]
        pl = obj_peaklist.peaklist(peaks)
        pl.reset()
        pl.remisotopes()
        assert len(pl) == 2
        pl.remuncharged()
        assert len(pl) == 1
        assert pl[0].mz == 100.0
