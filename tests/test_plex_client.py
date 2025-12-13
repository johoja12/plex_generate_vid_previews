"""
Tests for plex_client.py module.

Tests Plex server connection, retry logic, library queries,
and duplicate filtering.
"""

import pytest
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch
import requests

from plex_generate_previews.plex_client import (
    plex_server,
    retry_plex_call,
    filter_duplicate_locations,
    get_library_sections
)


class TestPlexServerConnection:
    """Test Plex server connection."""
    
    @patch('plexapi.server.PlexServer')
    @patch('requests.Session')
    def test_plex_server_connection_success(self, mock_session, mock_plex_server, mock_config):
        """Test successful connection to Plex server."""
        mock_plex = MagicMock()
        mock_plex_server.return_value = mock_plex
        
        result = plex_server(mock_config)
        
        assert result == mock_plex
        mock_plex_server.assert_called_once()
    
    @patch('plexapi.server.PlexServer')
    @patch('requests.Session')
    def test_plex_server_connection_failure(self, mock_session, mock_plex_server, mock_config):
        """Test connection error handling."""
        mock_plex_server.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with pytest.raises(ConnectionError):
            plex_server(mock_config)
    
    @patch('plexapi.server.PlexServer')
    @patch('requests.Session')
    def test_plex_server_timeout(self, mock_session, mock_plex_server, mock_config):
        """Test timeout handling."""
        mock_plex_server.side_effect = requests.exceptions.ReadTimeout("Timeout")
        
        with pytest.raises(ConnectionError):
            plex_server(mock_config)
    
    @patch('plexapi.server.PlexServer')
    @patch('requests.Session')
    def test_plex_server_retry_strategy(self, mock_session, mock_plex_server, mock_config):
        """Test that retry strategy is configured."""
        mock_plex = MagicMock()
        mock_plex_server.return_value = mock_plex
        
        result = plex_server(mock_config)
        
        # Verify session was configured with retry
        assert mock_session.called
        session = mock_session.return_value
        assert session.mount.called


class TestRetryPlexCall:
    """Test Plex API retry logic."""
    
    def test_retry_plex_call_success(self):
        """Test call succeeds on first try."""
        mock_func = MagicMock(return_value="success")
        
        result = retry_plex_call(mock_func, "arg1", "arg2", kwarg1="value1")
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")
    
    def test_retry_plex_call_xml_error_retry(self):
        """Test retry on XML parse error."""
        mock_func = MagicMock()
        # Fail first time, succeed second time
        mock_func.side_effect = [
            ET.ParseError("syntax error"),
            "success"
        ]
        
        result = retry_plex_call(mock_func, max_retries=2, retry_delay=0.01)
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_retry_plex_call_max_retries(self):
        """Test give up after max retries."""
        mock_func = MagicMock()
        mock_func.side_effect = ET.ParseError("syntax error")
        
        with pytest.raises(ET.ParseError):
            retry_plex_call(mock_func, max_retries=2, retry_delay=0.01)
        
        # Should try 3 times (initial + 2 retries)
        assert mock_func.call_count == 3
    
    def test_retry_plex_call_non_xml_error(self):
        """Test non-XML errors are not retried."""
        mock_func = MagicMock()
        mock_func.side_effect = ValueError("Some other error")
        
        with pytest.raises(ValueError):
            retry_plex_call(mock_func, max_retries=2)
        
        # Should only try once (no retry for non-XML errors)
        assert mock_func.call_count == 1


class TestFilterDuplicateLocations:
    """Test duplicate location filtering."""
    
    def test_filter_duplicate_locations(self):
        """Test basic duplicate filtering."""
        media_items = [
            ('key1', ['/path/to/video1.mkv'], 'Show S01E01', 'episode', 'hash1', 'added1'),
            ('key2', ['/path/to/video2.mkv'], 'Show S01E02', 'episode', 'hash2', 'added2'),
            ('key3', ['/path/to/video1.mkv'], 'Show S01E01 Part 2', 'episode', 'hash3', 'added3'),  # Duplicate!
        ]
        
        filtered = filter_duplicate_locations(media_items)
        
        # Should only have 2 items (duplicate removed)
        assert len(filtered) == 2
        assert ('key1', 'Show S01E01', 'episode', 'hash1', 'added1') in filtered
        assert ('key2', 'Show S01E02', 'episode', 'hash2', 'added2') in filtered
        assert ('key3', 'Show S01E01 Part 2', 'episode', 'hash3', 'added3') not in filtered
    
    def test_filter_duplicate_locations_multiple_files(self):
        """Test filtering with multi-part episodes."""
        media_items = [
            ('key1', ['/path/to/video1.mkv', '/path/to/video2.mkv'], 'Show S01E01-E02', 'episode', 'hash1', 'added1'),
            ('key2', ['/path/to/video2.mkv', '/path/to/video3.mkv'], 'Show S01E02-E03', 'episode', 'hash2', 'added2'),  # Overlaps!
        ]
        
        filtered = filter_duplicate_locations(media_items)
        
        # Second item should be filtered out due to overlap
        assert len(filtered) == 1
        assert ('key1', 'Show S01E01-E02', 'episode', 'hash1', 'added1') in filtered
    
    def test_filter_duplicate_locations_empty(self):
        """Test filtering empty list."""
        filtered = filter_duplicate_locations([])
        assert filtered == []
    
    def test_filter_duplicate_locations_no_duplicates(self):
        """Test filtering with no duplicates."""
        media_items = [
            ('key1', ['/path/to/video1.mkv'], 'Movie 1', 'movie', 'hash1', 'added1'),
            ('key2', ['/path/to/video2.mkv'], 'Movie 2', 'movie', 'hash2', 'added2'),
            ('key3', ['/path/to/video3.mkv'], 'Movie 3', 'movie', 'hash3', 'added3'),
        ]
        
        filtered = filter_duplicate_locations(media_items)
        
        assert len(filtered) == 3
        assert ('key1', 'Movie 1', 'movie', 'hash1', 'added1') in filtered
        assert ('key2', 'Movie 2', 'movie', 'hash2', 'added2') in filtered
        assert ('key3', 'Movie 3', 'movie', 'hash3', 'added3') in filtered


class TestGetLibrarySections:
    """Test library section retrieval."""
    
    @patch('plex_generate_previews.plex_client.get_hash_with_fallback')
    def test_get_library_sections_movies(self, mock_get_hash, mock_config):
        """Test getting movie libraries."""
        mock_get_hash.return_value = "somehashmovie"

        mock_plex = MagicMock()

        # Mock section
        mock_section = MagicMock()
        mock_section.title = "Movies"
        mock_section.METADATA_TYPE = "movie"

        # Mock movie
        mock_movie = MagicMock()
        mock_movie.key = "/library/metadata/1"
        mock_movie.ratingKey = "1"  # Plex API returns ratingKey as ID
        mock_movie.title = "Test Movie"
        mock_movie.locations = ["/data/movies/Test Movie.mkv"]
        mock_movie.addedAt = "2023-01-01T12:00:00Z"

        mock_section.search.return_value = [mock_movie]
        mock_plex.library.sections.return_value = [mock_section]

        sections = list(get_library_sections(mock_plex, mock_config))

        assert len(sections) == 1
        section, media = sections[0]
        assert section == mock_section
        assert len(media) == 1
        # Format: (ratingKey, title, type, bundle_hash, addedAt)
        assert media[0][0] == "1" # ratingKey
        assert media[0][1] == "Test Movie" # title
        assert media[0][2] == "movie" # type
        assert media[0][3] == "somehashmovie" # bundle_hash
        assert media[0][4] == "2023-01-01T12:00:00Z" # addedAt
    
    @patch('plex_generate_previews.plex_client.get_hash_with_fallback')
    def test_get_library_sections_episodes(self, mock_get_hash, mock_config):
        """Test getting TV show libraries."""
        mock_get_hash.return_value = "somehashepisode"

        mock_plex = MagicMock()

        # Mock section
        mock_section = MagicMock()
        mock_section.title = "TV Shows"
        mock_section.METADATA_TYPE = "episode"

        # Mock episode
        mock_episode = MagicMock()
        mock_episode.key = "/library/metadata/123"
        mock_episode.ratingKey = "123" # Plex API returns ratingKey as ID
        mock_episode.grandparentTitle = "Test Show"
        mock_episode.seasonEpisode = "s01e01"
        mock_episode.locations = ["/path/to/show.mkv"]
        mock_episode.addedAt = "2023-01-01T12:00:00Z"

        mock_section.search.return_value = [mock_episode]
        mock_plex.library.sections.return_value = [mock_section]

        sections = list(get_library_sections(mock_plex, mock_config))

        assert len(sections) == 1
        section, media = sections[0]
        assert section == mock_section
        assert len(media) == 1
        # Format after duplicate filtering: (ratingKey, title, type, bundle_hash, addedAt)
        assert media[0][0] == "123" # ratingKey
        assert "Test Show" in media[0][1] # title
        assert "S01E01" in media[0][1] # title
        assert media[0][2] == "episode" # type
        assert media[0][3] == "somehashepisode" # bundle_hash
        assert media[0][4] == "2023-01-01T12:00:00Z" # addedAt
    
    def test_get_library_sections_filter(self, mock_config):
        """Test filtering by configured libraries."""
        mock_config.plex_libraries = ['movies']
        
        mock_plex = MagicMock()
        
        # Mock two sections
        mock_section_movies = MagicMock()
        mock_section_movies.title = "Movies"
        mock_section_movies.METADATA_TYPE = "movie"
        
        mock_section_tv = MagicMock()
        mock_section_tv.title = "TV Shows"
        mock_section_tv.METADATA_TYPE = "episode"
        
        mock_plex.library.sections.return_value = [mock_section_movies, mock_section_tv]
        
        sections = list(get_library_sections(mock_plex, mock_config))
        
        # Should only include Movies
        assert len(sections) == 1
        assert sections[0][0].title == "Movies"
    
    def test_get_library_sections_unsupported(self, mock_config):
        """Test skipping unsupported library types."""
        mock_plex = MagicMock()
        
        # Mock photo section (unsupported)
        mock_section = MagicMock()
        mock_section.title = "Photos"
        mock_section.METADATA_TYPE = "photo"
        
        mock_plex.library.sections.return_value = [mock_section]
        
        sections = list(get_library_sections(mock_plex, mock_config))
        
        # Should skip photos
        assert len(sections) == 0
    
    def test_get_library_sections_api_error(self, mock_config):
        """Test handling of API errors."""
        mock_plex = MagicMock()
        mock_plex.library.sections.side_effect = requests.exceptions.RequestException("API error")
        
        sections = list(get_library_sections(mock_plex, mock_config))
        
        # Should handle error gracefully
        assert sections == []
    
    def test_get_library_sections_search_error(self, mock_config):
        """Test handling of search errors."""
        import requests
        mock_plex = MagicMock()
        
        mock_section = MagicMock()
        mock_section.title = "Movies"
        mock_section.METADATA_TYPE = "movie"
        # Use a specific exception type that's caught by the code
        mock_section.search.side_effect = requests.exceptions.RequestException("Search failed")
        
        mock_plex.library.sections.return_value = [mock_section]
        
        sections = list(get_library_sections(mock_plex, mock_config))
        
        # Should handle error and skip this library
        assert len(sections) == 0
    
    def test_get_library_sections_duplicate_filtering(self, mock_config):
        """Test that duplicate episodes are filtered."""
        mock_plex = MagicMock()
        
        mock_section = MagicMock()
        mock_section.title = "TV Shows"
        mock_section.METADATA_TYPE = "episode"
        
        # Two episodes pointing to same file (multi-part episode)
        mock_episode1 = MagicMock()
        mock_episode1.key = "/library/metadata/1"
        mock_episode1.grandparentTitle = "Show"
        mock_episode1.seasonEpisode = "s01e01"
        mock_episode1.locations = ["/path/to/episode.mkv"]
        
        mock_episode2 = MagicMock()
        mock_episode2.key = "/library/metadata/2"
        mock_episode2.grandparentTitle = "Show"
        mock_episode2.seasonEpisode = "s01e02"
        mock_episode2.locations = ["/path/to/episode.mkv"]  # Same file!
        
        mock_section.search.return_value = [mock_episode1, mock_episode2]
        mock_plex.library.sections.return_value = [mock_section]
        
        sections = list(get_library_sections(mock_plex, mock_config))
        
        # Should only have 1 episode (duplicate filtered)
        assert len(sections) == 1
        section, media = sections[0]
        assert len(media) == 1

