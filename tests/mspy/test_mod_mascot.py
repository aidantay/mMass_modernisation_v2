import pytest
import tempfile
import os
import mspy.mod_mascot


# Test data - Sample XML responses

SAMPLE_XML_SINGLE_HIT = '''<?xml version="1.0" encoding="UTF-8"?>
<mascot_search_results>
  <hit number="1">
    <protein accession="Q12345">
      <prot_desc>Test Protein Description</prot_desc>
      <prot_score>95.5</prot_score>
      <prot_mass>50000.12</prot_mass>
      <peptide query="1" rank="1" isbold="1">
        <pep_seq>PEPTIDER</pep_seq>
        <pep_score>45.3</pep_score>
        <pep_calc_mr>900.45</pep_calc_mr>
      </peptide>
    </protein>
  </hit>
</mascot_search_results>'''

SAMPLE_XML_MULTIPLE_HITS = '''<?xml version="1.0" encoding="UTF-8"?>
<mascot_search_results>
  <hit number="1">
    <protein accession="Q12345">
      <prot_desc>Test Protein 1</prot_desc>
      <prot_score>95.5</prot_score>
      <peptide query="1" rank="1" isbold="1">
        <pep_seq>PEPTIDER</pep_seq>
        <pep_score>45.3</pep_score>
      </peptide>
      <peptide query="2" rank="1" isbold="1">
        <pep_seq>PEPTIDE2</pep_seq>
        <pep_score>50.1</pep_score>
      </peptide>
    </protein>
    <protein accession="P54321">
      <prot_desc>Test Protein 2</prot_desc>
      <prot_score>75.2</prot_score>
      <peptide query="3" rank="1" isbold="0">
        <pep_seq>PROTEIN</pep_seq>
        <pep_score>30.5</pep_score>
      </peptide>
    </protein>
  </hit>
  <hit number="2">
    <protein accession="Q99999">
      <prot_desc>Test Protein 3</prot_desc>
      <prot_score>60.0</prot_score>
      <peptide query="4" rank="1" isbold="1">
        <pep_seq>SEQUENCE</pep_seq>
        <pep_score>35.7</pep_score>
      </peptide>
    </protein>
  </hit>
</mascot_search_results>'''

SAMPLE_XML_NO_HITS = '''<?xml version="1.0" encoding="UTF-8"?>
<mascot_search_results>
</mascot_search_results>'''

SAMPLE_PARAMETERS = '''[DB]
SwissProt
TrEMBL

[CLE]
Trypsin
Pepsin

[MODS]
Acetyl (N-term)
Oxidation (M)

[INSTRUMENT]
Default
ESI-Q-TOF

[TAXONOMY]
All entries
Homo sapiens

[HIDDEN_MODS]
Label:18O(2)

[OPTIONS]
None
Instrument settings

[QUANTITATION]
None
SILAC

'''

SAMPLE_PARAMETERS_WITH_EMPTY_LINES = '''[DB]
SwissProt

[CLE]
Trypsin

'''

SAMPLE_SEARCH_RESPONSE = '''<html>
<script>
  var uQuery='123';
  var uFG='2';
  var uErr=0;
  var uUrl="master_results.pl?file=F001234567&REPTYPE=peptide";
</script>
</html>'''

SAMPLE_SEARCH_RESPONSE_NO_MATCH = '''<html>
<body>Server error</body>
</html>'''


# ====== Tests: Initialization ======

class TestMascotInit(object):
    """Tests for mascot.__init__"""

    def test_init_default_path(self):
        """Test mascot initialization with default path."""
        m = mspy.mod_mascot.mascot('testhost.com')

        assert m.server['host'] == 'testhost.com'
        assert m.server['path'] == '/mascot/'
        assert m.server['search'] == 'cgi/nph-mascot.exe'
        assert m.server['result'] == 'cgi/master_results.pl'
        assert m.server['export'] == 'cgi/export_dat_2.pl'
        assert m.server['params'] == 'cgi/get_params.pl'
        assert m.resultsPath is None
        assert m.resultsXML is None
        assert m.hits == {}

    def test_init_custom_path(self):
        """Test mascot initialization with custom path."""
        m = mspy.mod_mascot.mascot('mascot.example.org', path='/search/mascot/')

        assert m.server['host'] == 'mascot.example.org'
        assert m.server['path'] == '/search/mascot/'

    def test_init_export_dict(self):
        """Test that export dict is initialized with correct values."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Check key export settings
        assert m.export['do_export'] == 1
        assert m.export['export_format'] == 'XML'
        assert m.export['report'] == 'AUTO'
        assert m.export['protein_master'] == 1
        assert m.export['peptide_master'] == 1
        assert m.export['_sigthreshold'] == 0.05
        assert m.export['prot_seq'] == 1


# ====== Tests: Search Method ======

class TestMascotSearch(object):
    """Tests for mascot.search"""

    def test_search_success_with_regex_match(self, mocker):
        """Test successful search with regex match in response."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Mock HTTPConnection
        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_SEARCH_RESPONSE

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.search('test query')

        assert result is True
        assert m.resultsPath == 'F001234567&REPTYPE=peptide'
        assert m.resultsXML is None
        assert m.hits == {}

    def test_search_network_exception(self, mocker):
        """Test search when network exception occurs."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Mock HTTPConnection to raise exception
        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', side_effect=Exception('Connection refused'))

        result = m.search('test query')

        assert result is False
        assert m.resultsPath is None
        assert m.resultsXML is None
        assert m.hits == {}

    def test_search_non_200_status(self, mocker):
        """Test search when server returns non-200 status."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_response = mocker.MagicMock()
        mock_response.status = 500
        mock_response.read.return_value = 'Server error'

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.search('test query')

        assert result is False

    def test_search_no_regex_match(self, mocker):
        """Test search when response does not contain regex match."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_SEARCH_RESPONSE_NO_MATCH

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.search('test query')

        assert result is False
        assert m.resultsPath is None

    def test_search_clears_previous_results(self, mocker):
        """Test that search clears previous results."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Set previous state
        m.resultsPath = 'old_path'
        m.resultsXML = 'old xml'
        m.hits = {1: {'protein': 'data'}}

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_SEARCH_RESPONSE

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        m.search('test query')

        # Previous resultsXML and hits should be cleared, but path set by search
        assert m.resultsXML is None
        assert m.hits == {}

    def test_search_multipart_body_format(self, mocker):
        """Test that search constructs proper multipart form data."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_SEARCH_RESPONSE

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        query_data = 'BEGIN IONS\nTITLE=test\nEND IONS'
        m.search(query_data)

        # Verify send was called with proper format
        mock_conn.send.assert_called_once()
        sent_body = mock_conn.send.call_args[0][0]
        assert 'Content-Disposition: form-data; name="QUE"' in sent_body
        assert query_data in sent_body

    def test_search_http_method_and_path(self, mocker):
        """Test that search uses correct HTTP method and path."""
        m = mspy.mod_mascot.mascot('testhost.com', path='/custom/')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_SEARCH_RESPONSE

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        m.search('test')

        # Verify putrequest was called with POST and correct path
        assert mock_conn.putrequest.called
        call_args = mock_conn.putrequest.call_args[0]
        assert call_args[0] == 'POST'
        assert '/custom/' in call_args[1]
        assert 'cgi/nph-mascot.exe' in call_args[1]


# ====== Tests: Report Method ======

class TestMascotReport(object):
    """Tests for mascot.report"""

    def test_report_with_explicit_path(self, mocker):
        """Test report with explicit path parameter."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_open = mocker.patch('mspy.mod_mascot.webbrowser.open')

        m.report(path='F001234567')

        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        assert 'http://' in url
        assert 'testhost.com' in url
        assert 'master_results.pl' in url
        assert 'F001234567' in url

    def test_report_with_self_results_path(self, mocker):
        """Test report using self.resultsPath."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'F001234567'

        mock_open = mocker.patch('mspy.mod_mascot.webbrowser.open')

        m.report()

        mock_open.assert_called_once()
        url = mock_open.call_args[0][0]
        assert 'F001234567' in url

    def test_report_no_path_silent_failure(self, mocker):
        """Test report when no path is available (silent failure)."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_open = mocker.patch('mspy.mod_mascot.webbrowser.open')

        # Both path and resultsPath are None
        m.report(path=None)

        # Should not call webbrowser.open
        assert not mock_open.called

    def test_report_explicit_path_takes_precedence(self, mocker):
        """Test that explicit path takes precedence over self.resultsPath."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'old_path'

        mock_open = mocker.patch('mspy.mod_mascot.webbrowser.open')

        m.report(path='new_path')

        url = mock_open.call_args[0][0]
        assert 'new_path' in url
        assert 'old_path' not in url


# ====== Tests: Fetchall Method ======

class TestMascotFetchall(object):
    """Tests for mascot.fetchall"""

    def test_fetchall_no_path_no_results_path(self):
        """Test fetchall returns False when no path available."""
        m = mspy.mod_mascot.mascot('testhost.com')

        result = m.fetchall()

        assert result is False

    def test_fetchall_with_explicit_path(self, mocker):
        """Test fetchall with explicit path parameter."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_XML_SINGLE_HIT

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.fetchall(path='F001234567')

        assert result is True
        assert m.resultsXML == SAMPLE_XML_SINGLE_HIT
        assert len(m.hits) > 0

    def test_fetchall_with_self_results_path(self, mocker):
        """Test fetchall using self.resultsPath."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'F001234567'

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_XML_SINGLE_HIT

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.fetchall()

        assert result is True
        assert m.resultsXML == SAMPLE_XML_SINGLE_HIT

    def test_fetchall_clears_previous_results(self, mocker):
        """Test fetchall clears previous results."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Set previous state
        m.resultsXML = 'old xml'
        m.hits = {1: {'protein': 'data'}}

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_XML_NO_HITS

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        m.fetchall(path='F001234567')

        # Should have new XML but empty hits
        assert m.resultsXML == SAMPLE_XML_NO_HITS
        assert m.hits == {}

    def test_fetchall_network_exception(self, mocker):
        """Test fetchall returns False on network exception."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'F001234567'

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', side_effect=Exception('Connection failed'))

        result = m.fetchall()

        assert result is False

    def test_fetchall_non_200_status(self, mocker):
        """Test fetchall returns False on non-200 status."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'F001234567'

        mock_response = mocker.MagicMock()
        mock_response.status = 404
        mock_response.read.return_value = 'Not found'

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.fetchall()

        assert result is False

    def test_fetchall_calls_parse(self, mocker):
        """Test fetchall calls parse method with XML data."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'F001234567'

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_XML_SINGLE_HIT

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.fetchall()

        # If parse was called successfully, hits should be populated
        assert len(m.hits) == 1
        assert 1 in m.hits

    def test_fetchall_includes_export_params(self, mocker):
        """Test fetchall includes all export parameters in request."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'F001234567'

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_XML_SINGLE_HIT

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        m.fetchall()

        # Verify request was called with export params
        mock_conn.request.assert_called_once()
        path = mock_conn.request.call_args[0][1]

        # Check key export parameters are included
        assert 'do_export=1' in path
        assert 'export_format=XML' in path
        assert 'protein_master=1' in path
        assert 'peptide_master=1' in path


# ====== Tests: Parse Method ======

class TestMascotParse(object):
    """Tests for mascot.parse"""

    def test_parse_from_string_data(self):
        """Test parse from string data parameter."""
        m = mspy.mod_mascot.mascot('testhost.com')

        result = m.parse(data=SAMPLE_XML_SINGLE_HIT)

        assert result is True
        assert len(m.hits) == 1
        assert 1 in m.hits
        assert 'Q12345' in m.hits[1]

    def test_parse_from_self_results_xml(self):
        """Test parse from self.resultsXML."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsXML = SAMPLE_XML_SINGLE_HIT

        result = m.parse()

        assert result is True
        assert len(m.hits) == 1

    def test_parse_from_file_path(self, mocker):
        """Test parse from file path parameter."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Create temporary file with XML content
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.xml') as f:
            f.write(SAMPLE_XML_SINGLE_HIT)
            temp_path = f.name

        try:
            result = m.parse(path=temp_path)

            assert result is True
            assert len(m.hits) == 1
        finally:
            os.unlink(temp_path)

    def test_parse_no_args_no_results_xml(self):
        """Test parse returns False when no arguments and no self.resultsXML."""
        m = mspy.mod_mascot.mascot('testhost.com')

        result = m.parse()

        assert result is False

    def test_parse_invalid_xml(self):
        """Test parse returns False on invalid XML."""
        m = mspy.mod_mascot.mascot('testhost.com')

        invalid_xml = '<mascot_search_results><unclosed>'
        result = m.parse(data=invalid_xml)

        assert result is False

    def test_parse_file_not_found(self):
        """Test parse returns False when file not found."""
        m = mspy.mod_mascot.mascot('testhost.com')

        result = m.parse(path='/nonexistent/path/to/file.xml')

        assert result is False

    def test_parse_no_hits(self):
        """Test parse with XML containing no hits."""
        m = mspy.mod_mascot.mascot('testhost.com')

        result = m.parse(data=SAMPLE_XML_NO_HITS)

        assert result is True
        assert m.hits == {}

    def test_parse_single_hit_single_protein_single_peptide(self):
        """Test parse extracts single hit with one protein and peptide."""
        m = mspy.mod_mascot.mascot('testhost.com')

        m.parse(data=SAMPLE_XML_SINGLE_HIT)

        assert 1 in m.hits
        assert 'Q12345' in m.hits[1]

        protein = m.hits[1]['Q12345']
        assert protein['prot_accession'] == 'Q12345'
        assert protein['prot_desc'] == 'Test Protein Description'
        assert protein['prot_score'] == '95.5'
        assert protein['prot_mass'] == '50000.12'
        assert len(protein['peptides']) == 1

        peptide = protein['peptides'][0]
        assert peptide['query'] == '1'
        assert peptide['rank'] == '1'
        assert peptide['isbold'] == '1'
        assert peptide['pep_seq'] == 'PEPTIDER'
        assert peptide['pep_score'] == '45.3'

    def test_parse_multiple_hits_multiple_proteins_multiple_peptides(self):
        """Test parse with multiple hits, proteins, and peptides.

        Note: Due to XML DOM getElementsByTagName behavior, all peptides
        within a hit are assigned to each protein in that hit.
        """
        m = mspy.mod_mascot.mascot('testhost.com')

        m.parse(data=SAMPLE_XML_MULTIPLE_HITS)

        # Check hit 1 has 2 proteins
        assert 1 in m.hits
        assert len(m.hits[1]) == 2
        assert 'Q12345' in m.hits[1]
        assert 'P54321' in m.hits[1]

        # Check first protein has all peptides from hit 1 (3 total)
        protein1 = m.hits[1]['Q12345']
        assert len(protein1['peptides']) == 3

        # Check second protein also has all peptides from hit 1
        protein2 = m.hits[1]['P54321']
        assert len(protein2['peptides']) == 3

        # Check hit 2
        assert 2 in m.hits
        assert 'Q99999' in m.hits[2]

    def test_parse_clears_previous_hits(self):
        """Test parse clears previous hits when called with new data."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Parse first XML
        m.parse(data=SAMPLE_XML_MULTIPLE_HITS)
        assert len(m.hits) == 2

        # Parse second XML with no hits
        m.parse(data=SAMPLE_XML_NO_HITS)

        # Hits should be empty now
        assert m.hits == {}

    def test_parse_clears_results_path_when_path_param(self):
        """Test parse clears resultsPath when called with path parameter."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'old_path'
        m.resultsXML = 'old xml'

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.xml') as f:
            f.write(SAMPLE_XML_SINGLE_HIT)
            temp_path = f.name

        try:
            m.parse(path=temp_path)

            assert m.resultsPath is None
            assert m.resultsXML is None
        finally:
            os.unlink(temp_path)

    def test_parse_clears_results_path_when_data_param(self):
        """Test parse clears resultsPath when called with data parameter."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'old_path'
        m.resultsXML = 'old xml'

        m.parse(data=SAMPLE_XML_SINGLE_HIT)

        assert m.resultsPath is None
        assert m.resultsXML is None

    def test_parse_handles_text_nodes_in_elements(self):
        """Test parse handles text nodes and only processes element nodes."""
        xml_with_text = '''<?xml version="1.0" encoding="UTF-8"?>
<mascot_search_results>
  <hit number="1">
    <protein accession="Q12345">
      <prot_desc>Test Protein</prot_desc>
      Text content that should be ignored
      <prot_score>95.5</prot_score>
    </protein>
  </hit>
</mascot_search_results>'''

        m = mspy.mod_mascot.mascot('testhost.com')
        result = m.parse(data=xml_with_text)

        assert result is True
        assert 1 in m.hits
        protein = m.hits[1]['Q12345']
        assert protein['prot_desc'] == 'Test Protein'
        assert protein['prot_score'] == '95.5'

    def test_parse_attributes_correctly_extracted(self):
        """Test that attributes like query, rank, isbold are extracted correctly.

        Note: All peptides in a hit are assigned to each protein in that hit.
        """
        m = mspy.mod_mascot.mascot('testhost.com')

        m.parse(data=SAMPLE_XML_MULTIPLE_HITS)

        protein = m.hits[1]['Q12345']

        # Check peptides are present with correct attributes
        # Due to getElementsByTagName, all 3 peptides from hit 1 are in each protein
        assert len(protein['peptides']) == 3

        # Find peptide with query=1, rank=1, isbold=1
        pep1 = protein['peptides'][0]
        assert pep1['query'] == '1'
        assert pep1['rank'] == '1'
        assert pep1['isbold'] == '1'

        # Find peptide with query=2
        pep2 = protein['peptides'][1]
        assert pep2['query'] == '2'
        assert pep2['rank'] == '1'
        assert pep2['isbold'] == '1'

        # Find peptide with isbold=0 (query=3)
        pep3 = protein['peptides'][2]
        assert pep3['isbold'] == '0'


# ====== Tests: Save Method ======

class TestMascotSave(object):
    """Tests for mascot.save"""

    def test_save_no_results_xml(self):
        """Test save returns False when no resultsXML."""
        m = mspy.mod_mascot.mascot('testhost.com')

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            result = m.save(temp_path)
            assert result is False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_successful_write(self):
        """Test successful save of results XML to file."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsXML = SAMPLE_XML_SINGLE_HIT

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            result = m.save(temp_path)

            assert result is True

            # Verify file contents
            with open(temp_path, 'r') as f:
                content = f.read()

            # In Python 2.7, the content should be the UTF-8 encoded version
            assert 'mascot_search_results' in content or 'mascot_search_results' in content.decode('utf-8', errors='ignore')
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_io_error(self, mocker):
        """Test save returns False on IOError."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsXML = SAMPLE_XML_SINGLE_HIT

        # Mock file() to raise IOError
        mocker.patch('__builtin__.open', side_effect=IOError('Permission denied'))

        result = m.save('/some/path/file.xml')

        assert result is False

    def test_save_file_builtin_error(self, mocker):
        """Test save handles file builtin errors."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsXML = SAMPLE_XML_SINGLE_HIT

        # Mock the file builtin (which save() uses)
        mocker.patch('__builtin__.open', side_effect=IOError('Disk full'))

        result = m.save('/nonexistent/path/file.xml')

        assert result is False


# ====== Tests: Parameters Method ======

class TestMascotParameters(object):
    """Tests for mascot.parameters"""

    def test_parameters_success_basic(self, mocker):
        """Test successful parameters retrieval."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_PARAMETERS

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.parameters()

        assert isinstance(result, dict)
        assert 'DB' in result
        assert 'CLE' in result
        assert 'MODS' in result
        assert 'SwissProt' in result['DB']
        assert 'TrEMBL' in result['DB']

    def test_parameters_network_exception(self, mocker):
        """Test parameters returns False on network exception."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', side_effect=Exception('Connection error'))

        result = m.parameters()

        assert result is False

    def test_parameters_default_sections_exist(self, mocker):
        """Test that parameters dict contains pre-populated default sections."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Empty response - should still have defaults
        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = ''

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.parameters()

        assert 'TAXONOMY' in result
        assert 'INSTRUMENT' in result
        assert 'QUANTITATION' in result
        assert 'OPTIONS' in result
        assert 'All entries' in result['TAXONOMY']
        assert 'Default' in result['INSTRUMENT']
        assert 'None' in result['QUANTITATION']
        assert 'None' in result['OPTIONS']

    def test_parameters_empty_lines_skipped(self, mocker):
        """Test that empty lines in parameter file are skipped."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_PARAMETERS_WITH_EMPTY_LINES

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.parameters()

        # Should have proper sections with no empty entries
        assert 'DB' in result
        assert 'SwissProt' in result['DB']
        assert '' not in result['DB']

    def test_parameters_multiple_entries_per_section(self, mocker):
        """Test parameters parses multiple entries per section."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_PARAMETERS

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.parameters()

        # CLE section should have Trypsin and Pepsin
        assert len(result['CLE']) >= 2
        assert 'Trypsin' in result['CLE']
        assert 'Pepsin' in result['CLE']

        # MODS section should have multiple modifications
        assert len(result['MODS']) >= 2
        assert 'Acetyl (N-term)' in result['MODS']
        assert 'Oxidation (M)' in result['MODS']

    def test_parameters_section_header_regex(self, mocker):
        """Test parameters correctly identifies section headers."""
        m = mspy.mod_mascot.mascot('testhost.com')

        param_text = '''[VALID_SECTION_NAME]
Entry1
Entry2
[ANOTHER_SECTION]
Entry3'''

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = param_text

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        result = m.parameters()

        assert 'VALID_SECTION_NAME' in result
        assert 'ANOTHER_SECTION' in result
        assert 'Entry1' in result['VALID_SECTION_NAME']
        assert 'Entry3' in result['ANOTHER_SECTION']

    def test_parameters_uses_get_method(self, mocker):
        """Test parameters uses GET method."""
        m = mspy.mod_mascot.mascot('testhost.com')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_PARAMETERS

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        m.parameters()

        # Verify request was called with GET method
        mock_conn.request.assert_called_once()
        assert mock_conn.request.call_args[0][0] == 'GET'

    def test_parameters_correct_url_path(self, mocker):
        """Test parameters uses correct URL path."""
        m = mspy.mod_mascot.mascot('testhost.com', path='/custom/')

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_PARAMETERS

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        m.parameters()

        # Verify correct path was used
        path = mock_conn.request.call_args[0][1]
        assert '/custom/' in path
        assert 'get_params.pl' in path


# ====== Integration Tests ======

class TestMascotIntegration(object):
    """Integration tests combining multiple methods."""

    def test_search_and_fetchall_workflow(self, mocker):
        """Test typical workflow of search followed by fetchall."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Mock search response
        search_response = mocker.MagicMock()
        search_response.status = 200
        search_response.read.return_value = SAMPLE_SEARCH_RESPONSE

        # Mock fetchall response
        fetchall_response = mocker.MagicMock()
        fetchall_response.status = 200
        fetchall_response.read.return_value = SAMPLE_XML_SINGLE_HIT

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.side_effect = [search_response, fetchall_response]

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        # Perform search
        search_result = m.search('test query')
        assert search_result is True

        # Fetch results
        fetchall_result = m.fetchall()
        assert fetchall_result is True

        # Verify hits are populated
        assert len(m.hits) > 0

    def test_fetchall_parse_save_workflow(self, mocker):
        """Test workflow of fetchall, parse, and save."""
        m = mspy.mod_mascot.mascot('testhost.com')
        m.resultsPath = 'F001234567'

        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = SAMPLE_XML_SINGLE_HIT

        mock_conn = mocker.MagicMock()
        mock_conn.getresponse.return_value = mock_response

        mocker.patch('mspy.mod_mascot.httplib.HTTPConnection', return_value=mock_conn)

        # Fetch results
        fetchall_result = m.fetchall()
        assert fetchall_result is True

        # Save to file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            save_result = m.save(temp_path)
            assert save_result is True

            # Verify file exists and has content
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
            assert len(content) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_parse_then_report_workflow(self, mocker):
        """Test workflow of parsing data and then opening report."""
        m = mspy.mod_mascot.mascot('testhost.com')

        # Parse XML
        m.parse(data=SAMPLE_XML_SINGLE_HIT)
        assert len(m.hits) > 0

        # Mock webbrowser
        mock_open = mocker.patch('mspy.mod_mascot.webbrowser.open')

        # Report with explicit path
        m.report(path='F001234567')

        assert mock_open.called
