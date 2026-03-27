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
import os
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, HealthCheck

try:
    from unittest.mock import patch, MagicMock, call
except ImportError:
    from mock import patch, MagicMock, call

import mod_utils
import mod_stopper


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def reset_stopper():
    """Reset stopper state before/after module tests."""
    mod_stopper.start()
    yield
    mod_stopper.start()


@pytest.fixture
def tmp_dir():
    """Create and cleanup a temporary directory."""
    tmp_path = tempfile.mkdtemp()
    yield tmp_path
    shutil.rmtree(tmp_path)


# ============================================================================
# TEST LOAD FUNCTION
# ============================================================================

class TestLoad(object):
    """Tests for mod_utils.load function."""

    def test_load_raises_ioerror_for_missing_file(self):
        """Test that load raises IOError for a file that doesn't exist."""
        non_existent_path = "/path/to/nonexistent/file.mzdata"
        with pytest.raises(IOError):
            mod_utils.load(non_existent_path)

    def test_load_mzdata_extension_real_file(self):
        """Test load with a real .mzdata file."""
        test_file = "/home/aidantay/projects/mMass_modernisation_v2/tests/data/test_small.mzdata"
        if not os.path.exists(test_file):
            pytest.skip("Test data file not available")
        result = mod_utils.load(test_file)
        # Parser returns False if no data, or a scan object otherwise
        assert result is not None

    def test_load_mzxml_extension_real_file(self):
        """Test load with a real .mzxml file."""
        test_file = "/home/aidantay/projects/mMass_modernisation_v2/tests/data/test_small.mzXML"
        if not os.path.exists(test_file):
            pytest.skip("Test data file not available")
        # Use specific scanID to avoid parsing entire file
        result = mod_utils.load(test_file, scanID=1)
        assert result is not None

    def test_load_mzml_extension_real_file(self):
        """Test load with a real .mzml file."""
        test_file = "/home/aidantay/projects/mMass_modernisation_v2/tests/data/test_small.mzML"
        if not os.path.exists(test_file):
            pytest.skip("Test data file not available")
        result = mod_utils.load(test_file, scanID=1)
        assert result is not None

    def test_load_mgf_extension_real_file(self):
        """Test load with a real .mgf file."""
        test_file = "/home/aidantay/projects/mMass_modernisation_v2/tests/data/test_tiny.mgf"
        if not os.path.exists(test_file):
            pytest.skip("Test data file not available")
        result = mod_utils.load(test_file)
        assert result is not None

    def test_load_xy_extension_continuous(self):
        """Test load with .xy file in continuous mode."""
        test_file = "/home/aidantay/projects/mMass_modernisation_v2/tests/data/test_small.xy"
        if not os.path.exists(test_file):
            pytest.skip("Test data file not available")
        result = mod_utils.load(test_file, dataType='continuous')
        assert result is not None

    def test_load_xy_extension_discrete(self):
        """Test load with .xy file in discrete mode."""
        test_file = "/home/aidantay/projects/mMass_modernisation_v2/tests/data/test_small.xy"
        if not os.path.exists(test_file):
            pytest.skip("Test data file not available")
        result = mod_utils.load(test_file, dataType='discrete')
        assert result is not None

    def test_load_txt_extension_mocked(self, tmp_dir):
        """Test load with .txt file using mocked parser."""
        # Create a real .txt file
        txt_path = os.path.join(tmp_dir, "test.txt")
        with open(txt_path, 'w') as f:
            f.write("100.0\t1000.0\n")
            f.write("200.0\t2000.0\n")

        with patch('mod_utils.parseXY') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(txt_path, dataType='continuous')

            # Verify parser was created and scan called
            mock_parser_class.assert_called_once_with(txt_path)
            mock_instance.scan.assert_called_once_with('continuous')
            assert result == mock_scan

    def test_load_asc_extension_mocked(self, tmp_dir):
        """Test load with .asc file using mocked parser."""
        asc_path = os.path.join(tmp_dir, "test.asc")
        with open(asc_path, 'w') as f:
            f.write("100.0\t1000.0\n")

        with patch('mod_utils.parseXY') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(asc_path, dataType='discrete')

            mock_parser_class.assert_called_once_with(asc_path)
            mock_instance.scan.assert_called_once_with('discrete')
            assert result == mock_scan

    def test_load_xml_sniff_mzdata(self, tmp_dir):
        """Test load with XML file containing mzData marker."""
        xml_path = os.path.join(tmp_dir, "test.xml")
        with open(xml_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n<mzData xmlns="http://psi.hupo.org/ms/mzdata">\n')

        with patch('mod_utils.parseMZDATA') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(xml_path, scanID=1)

            mock_parser_class.assert_called_once_with(xml_path)
            mock_instance.scan.assert_called_once_with(1)
            assert result == mock_scan

    def test_load_xml_sniff_mzxml(self, tmp_dir):
        """Test load with XML file containing mzXML marker."""
        xml_path = os.path.join(tmp_dir, "test.xml")
        with open(xml_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision">\n')

        with patch('mod_utils.parseMZXML') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(xml_path, scanID=2)

            mock_parser_class.assert_called_once_with(xml_path)
            mock_instance.scan.assert_called_once_with(2)
            assert result == mock_scan

    def test_load_xml_sniff_mzml(self, tmp_dir):
        """Test load with XML file containing mzML marker."""
        xml_path = os.path.join(tmp_dir, "test.xml")
        with open(xml_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n<mzML xmlns="http://psi.hupo.org/ms/mzml">\n')

        with patch('mod_utils.parseMZML') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(xml_path, scanID=3)

            mock_parser_class.assert_called_once_with(xml_path)
            mock_instance.scan.assert_called_once_with(3)
            assert result == mock_scan

    def test_load_xml_no_marker_raises_nameerror(self, tmp_dir):
        """Test load with XML file that has no recognized marker raises NameError."""
        xml_path = os.path.join(tmp_dir, "test.xml")
        with open(xml_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n<unknown></unknown>\n')

        # In Python 2, accessing undefined docType raises NameError, not ValueError
        with pytest.raises(NameError):
            mod_utils.load(xml_path)

    def test_load_unknown_extension_raises_nameerror(self, tmp_dir):
        """Test load with unknown file extension raises NameError."""
        unknown_path = os.path.join(tmp_dir, "test.unknown")
        with open(unknown_path, 'w') as f:
            f.write("dummy content")

        # In Python 2, accessing undefined docType raises NameError, not ValueError
        with pytest.raises(NameError):
            mod_utils.load(unknown_path)

    def test_load_extension_case_insensitive_uppercase(self, tmp_dir):
        """Test load handles uppercase extension properly."""
        xy_path = os.path.join(tmp_dir, "test.XY")
        with open(xy_path, 'w') as f:
            f.write("100.0\t1000.0\n")

        with patch('mod_utils.parseXY') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(xy_path)

            mock_parser_class.assert_called_once()
            assert result == mock_scan

    def test_load_mzdata_passes_scanid_to_parser(self, tmp_dir):
        """Test that load passes scanID to parser.scan()."""
        mzdata_path = os.path.join(tmp_dir, "test.mzdata")
        with open(mzdata_path, 'w') as f:
            f.write("<mzData></mzData>")

        with patch('mod_utils.parseMZDATA') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(mzdata_path, scanID=42)

            mock_instance.scan.assert_called_once_with(42)

    def test_load_xy_passes_datatype_to_parser(self, tmp_dir):
        """Test that load passes dataType to parser.scan()."""
        xy_path = os.path.join(tmp_dir, "test.xy")
        with open(xy_path, 'w') as f:
            f.write("100.0\t1000.0\n")

        with patch('mod_utils.parseXY') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(xy_path, dataType='discrete')

            mock_instance.scan.assert_called_once_with('discrete')

    def test_load_mgf_ignores_datatype(self, tmp_dir):
        """Test that load calls parser.scan(scanID) for MGF, ignoring dataType."""
        mgf_path = os.path.join(tmp_dir, "test.mgf")
        with open(mgf_path, 'w') as f:
            f.write("BEGIN IONS\nEND IONS\n")

        with patch('mod_utils.parseMGF') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(mgf_path, scanID=5, dataType='continuous')

            # MGF parser should receive scanID, not dataType
            mock_instance.scan.assert_called_once_with(5)

    def test_load_mzxml_parser_receives_scanid(self, tmp_dir):
        """Test that load passes scanID to mzXML parser."""
        mzxml_path = os.path.join(tmp_dir, "test.mzxml")
        with open(mzxml_path, 'w') as f:
            f.write("<mzXML></mzXML>")

        with patch('mod_utils.parseMZXML') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(mzxml_path, scanID=10)

            mock_instance.scan.assert_called_once_with(10)

    def test_load_mzml_parser_receives_scanid(self, tmp_dir):
        """Test that load passes scanID to mzML parser."""
        mzml_path = os.path.join(tmp_dir, "test.mzml")
        with open(mzml_path, 'w') as f:
            f.write("<mzML></mzML>")

        with patch('mod_utils.parseMZML') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            result = mod_utils.load(mzml_path, scanID=15)

            mock_instance.scan.assert_called_once_with(15)

    def test_load_returns_scan_object_from_parser(self, tmp_dir):
        """Test that load returns the scan object from the parser."""
        mzdata_path = os.path.join(tmp_dir, "test.mzdata")
        with open(mzdata_path, 'w') as f:
            f.write("<mzData></mzData>")

        mock_scan_obj = {'data': 'mock_scan'}
        with patch('mod_utils.parseMZDATA') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_instance.scan.return_value = mock_scan_obj

            result = mod_utils.load(mzdata_path)

            assert result == mock_scan_obj

    def test_load_parser_returns_false_propagates(self, tmp_dir):
        """Test that load returns False when parser returns False."""
        xy_path = os.path.join(tmp_dir, "test.xy")
        with open(xy_path, 'w') as f:
            f.write("invalid_data")

        with patch('mod_utils.parseXY') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_instance.scan.return_value = False

            result = mod_utils.load(xy_path)

            assert result == False

    def test_load_default_scanid_is_none(self, tmp_dir):
        """Test that load uses None as default scanID when not provided."""
        mzxml_path = os.path.join(tmp_dir, "test.mzxml")
        with open(mzxml_path, 'w') as f:
            f.write("<mzXML></mzXML>")

        with patch('mod_utils.parseMZXML') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            # Call without scanID parameter
            result = mod_utils.load(mzxml_path)

            # Should be called with None
            mock_instance.scan.assert_called_once_with(None)

    def test_load_default_datatype_is_continuous(self, tmp_dir):
        """Test that load uses 'continuous' as default dataType when not provided."""
        xy_path = os.path.join(tmp_dir, "test.xy")
        with open(xy_path, 'w') as f:
            f.write("100.0\t1000.0\n")

        with patch('mod_utils.parseXY') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan = MagicMock()
            mock_instance.scan.return_value = mock_scan

            # Call without dataType parameter
            result = mod_utils.load(xy_path)

            # Should be called with 'continuous'
            mock_instance.scan.assert_called_once_with('continuous')

    def test_load_mgf_with_scanid(self, tmp_dir):
        """Test that MGF parser receives scanID parameter."""
        mgf_path = os.path.join(tmp_dir, "test.mgf")
        with open(mgf_path, 'w') as f:
            f.write("BEGIN IONS\nTITLE=Test\nEND IONS\n")

        with patch('mod_utils.parseMGF') as mock_parser_class:
            mock_instance = MagicMock()
            mock_parser_class.return_value = mock_instance
            mock_scan_obj = {'ms_level': 2}
            mock_instance.scan.return_value = mock_scan_obj

            result = mod_utils.load(mgf_path, scanID=100)

            mock_instance.scan.assert_called_once_with(100)
            assert result == mock_scan_obj

    def test_load_mzdata_real_returns_nonempty(self):
        """Test load with real mzdata file returns something."""
        test_file = "/home/aidantay/projects/mMass_modernisation_v2/tests/data/test_small.mzdata"
        if not os.path.exists(test_file):
            pytest.skip("Test data file not available")

        result = mod_utils.load(test_file, scanID=1)
        # Should not be None for a real file with data
        assert result is not None


# ============================================================================
# TEST SAVE FUNCTION
# ============================================================================

class TestSave(object):
    """Tests for mod_utils.save function."""

    def test_save_nonempty_data_writes_correct_content(self, tmp_dir):
        """Test save with non-empty data writes correct format."""
        save_path = os.path.join(tmp_dir, "output.txt")
        data = [[100.0, 1000.0], [200.5, 2000.5], [300.123, 3000.789]]

        mod_utils.save(data, save_path)

        # Read back and verify
        assert os.path.exists(save_path)
        with open(save_path, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        assert len(lines) == 3

        # Check format is correct (floats with tabs)
        for i, line in enumerate(lines):
            parts = line.split('\t')
            assert len(parts) == 2
            # Values should be floats
            float(parts[0])
            float(parts[1])

    def test_save_empty_data_writes_empty_file(self, tmp_dir):
        """Test save with empty data list writes empty file."""
        save_path = os.path.join(tmp_dir, "empty.txt")
        data = []

        mod_utils.save(data, save_path)

        assert os.path.exists(save_path)
        with open(save_path, 'r') as f:
            content = f.read()
        assert content == ''

    def test_save_single_point(self, tmp_dir):
        """Test save with single data point."""
        save_path = os.path.join(tmp_dir, "single.txt")
        data = [[123.45, 678.90]]

        mod_utils.save(data, save_path)

        with open(save_path, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        assert len(lines) == 1
        assert '\t' in lines[0]

    def test_save_float_format_precision(self, tmp_dir):
        """Test that save preserves float precision."""
        save_path = os.path.join(tmp_dir, "precision.txt")
        # Use specific values to test precision
        data = [[100.123456, 200.987654]]

        mod_utils.save(data, save_path)

        with open(save_path, 'r') as f:
            content = f.read()

        # Python %f format defaults to 6 decimal places
        assert '100.123456' in content or '100.123456' in content.replace('\n', '')

    def test_save_large_numbers(self, tmp_dir):
        """Test save with large floating point numbers."""
        save_path = os.path.join(tmp_dir, "large.txt")
        data = [[1e6, 1e7], [1e8, 1e9]]

        mod_utils.save(data, save_path)

        with open(save_path, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        assert len(lines) == 2

    def test_save_zero_values(self, tmp_dir):
        """Test save with zero values."""
        save_path = os.path.join(tmp_dir, "zeros.txt")
        data = [[0.0, 0.0], [0.0, 100.0]]

        mod_utils.save(data, save_path)

        with open(save_path, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        assert len(lines) == 2
        assert '0.' in lines[0]

    def test_save_negative_values(self, tmp_dir):
        """Test save with negative values."""
        save_path = os.path.join(tmp_dir, "negative.txt")
        data = [[-100.0, -200.0], [100.0, 200.0]]

        mod_utils.save(data, save_path)

        with open(save_path, 'r') as f:
            content = f.read()

        assert '-100.' in content
        assert '-200.' in content

    def test_save_line_count_matches_data_length_property(self, tmp_dir):
        """Property-based test: save produces line count equal to data length."""
        @given(st.lists(st.lists(st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10
        ), min_size=2, max_size=2), max_size=20))
        @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
        def check_save_line_count(data):
            if not data:
                # Empty data case
                save_path = os.path.join(tmp_dir, "property_test_{}.txt".format(id(data)))
                mod_utils.save(data, save_path)
                with open(save_path, 'r') as f:
                    content = f.read()
                assert content == ''
            else:
                save_path = os.path.join(tmp_dir, "property_test_{}.txt".format(id(data)))
                mod_utils.save(data, save_path)
                with open(save_path, 'r') as f:
                    content = f.read()

                # Count non-empty lines
                if content.strip():
                    lines = content.strip().split('\n')
                    assert len(lines) == len(data)
                else:
                    assert len(data) == 0

        check_save_line_count()

    def test_save_overwrites_existing_file(self, tmp_dir):
        """Test that save overwrites existing file."""
        save_path = os.path.join(tmp_dir, "overwrite.txt")

        # Write initial content
        data1 = [[100.0, 200.0], [300.0, 400.0]]
        mod_utils.save(data1, save_path)

        # Overwrite with new content
        data2 = [[50.0, 60.0]]
        mod_utils.save(data2, save_path)

        with open(save_path, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        assert len(lines) == 1

    def test_save_with_very_small_numbers(self, tmp_dir):
        """Test save with very small numbers."""
        save_path = os.path.join(tmp_dir, "small_nums.txt")
        data = [[1e-6, 2e-6], [3e-7, 4e-7]]

        mod_utils.save(data, save_path)

        with open(save_path, 'r') as f:
            content = f.read()

        assert len(content) > 0
        lines = content.strip().split('\n')
        assert len(lines) == 2
