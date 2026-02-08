"""
Plex Media Server client and API interactions.

Handles Plex server connection, XML parsing monkey patch for debugging,
library querying, and duplicate location filtering.
"""

import os
import time
import http.client
import xml.etree.ElementTree
import requests
import urllib3
import sqlite3
from datetime import datetime
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from .config import Config


def retry_plex_call(func, *args, max_retries=3, retry_delay=1.0, **kwargs):
    """
    Retry a Plex API call if it fails due to XML parsing errors.
    
    This handles cases where Plex returns incomplete XML due to being busy.
    
    Args:
        func: Function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retries (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except xml.etree.ElementTree.ParseError as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"XML parsing error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                logger.info(f"Retrying in {retry_delay} seconds... (Plex may be busy)")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(f"XML parsing failed after {max_retries + 1} attempts: {e}")
        except Exception as e:
            # For non-XML errors, don't retry
            raise e
    
    # If we get here, all retries failed
    raise last_exception


def plex_server(config: Config):
    """
    Create Plex server connection with retry strategy and XML debugging.
    
    Args:
        config: Configuration object
        
    Returns:
        PlexServer: Configured Plex server instance
        
    Raises:
        ConnectionError: If unable to connect to Plex server
        requests.exceptions.RequestException: If connection fails after retries
    """
    # Plex Interface with retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.verify = False
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Create Plex server instance with proper error handling
    from plexapi.server import PlexServer
    try:
        logger.info(f"Connecting to Plex server at {config.plex_url}...")
        plex = PlexServer(config.plex_url, config.plex_token, timeout=config.plex_timeout, session=session)
        logger.info("Successfully connected to Plex server")
        return plex
    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout, 
            requests.exceptions.ReadTimeout, requests.exceptions.RequestException) as e:
        logger.error(f"Failed to connect to Plex server at {config.plex_url}")
        logger.error(f"Connection error: {e}")
        logger.error("Please check:")
        logger.error("  - Plex server is running and accessible")
        logger.error("  - Plex URL is correct (including http:// or https://)")
        logger.error("  - Network connectivity to Plex server")
        logger.error("  - Firewall settings allow connections to port 32400")
        raise ConnectionError(f"Unable to connect to Plex server at {config.plex_url}: {e}") from e


def filter_duplicate_locations(media_items):
    """
    Filter out duplicate media items based on file locations.
    
    This function prevents processing the same video file multiple times
    when it appears in multiple episodes (common with multi-part episodes).
    It keeps the first occurrence and skips subsequent duplicates.
    
    Args:
        media_items: List of tuples (key, locations, title, media_type, bundle_hash, added_at)
    
    Returns:
        list: Filtered list of tuples (key, title, media_type, bundle_hash, added_at) without duplicates
    """
    seen_locations = set()
    filtered_items = []
    
    for key, locations, title, media_type, bundle_hash, added_at in media_items:            
        # Check if any location has been seen before
        if any(location in seen_locations for location in locations):
            continue
            
        # Add all locations to seen set and keep this item
        seen_locations.update(locations)
        filtered_items.append((key, title, media_type, bundle_hash, added_at))
    
    return filtered_items


def get_first_hash(item):
    """Get the bundle hash of the first part of the first media."""
    try:
        return item.media[0].parts[0].hash
    except (IndexError, AttributeError):
        return None


def get_hash_with_fallback(item, plex_config_folder: str) -> Optional[str]:
    """
    Get the bundle hash directly from the Plex database.

    Queries the Plex database using the item ID for better performance
    and reliability compared to API responses.

    Note: The bundle hash is stored in media_parts.hash, not metadata_items.hash.
    The metadata_items.hash field is often NULL and serves a different purpose.

    Args:
        item: Plex media item object
        plex_config_folder: Path to Plex config folder

    Returns:
        str: Bundle hash if found, None otherwise
    """
    # Query database directly using item ID (more reliable than API)
    try:
        import sqlite3
        db_path = os.path.join(plex_config_folder, 'Plug-in Support', 'Databases', 'com.plexapp.plugins.library.db')

        if not os.path.exists(db_path):
            logger.debug(f"Plex database not found at: {db_path}")
            return None

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query media_parts.hash via JOIN (bundle hash is in media_parts, not metadata_items)
        query = """
            SELECT media_parts.hash
            FROM media_parts
            JOIN media_items ON media_parts.media_item_id = media_items.id
            WHERE media_items.metadata_item_id = ?
            LIMIT 1
        """
        cursor.execute(query, (item.ratingKey,))
        result = cursor.fetchone()

        conn.close()

        if result and result[0]:
            return result[0]
        else:
            logger.debug(f"No hash found in database for item {item.ratingKey} ({item.title})")
            return None

    except sqlite3.Error as e:
        logger.error(f"Database error while querying hash for {item.title}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error querying database for {item.title}: {type(e).__name__}: {e}")
        return None


def get_media_parts_from_database(plex_config_folder: str, rating_key: int):
    """
    Query media parts (file paths and bundle hashes) directly from the Plex SQLite database.

    This eliminates the need for Plex API calls during processing, making the system
    more resilient to Plex server outages and faster overall.

    Args:
        plex_config_folder: Path to Plex config folder
        rating_key: Plex rating key (item ID)

    Returns:
        list: List of tuples (file_path, bundle_hash) for each media part, or empty list if not found
    """
    db_path = os.path.join(plex_config_folder, 'Plug-in Support', 'Databases',
                           'com.plexapp.plugins.library.db')

    if not os.path.exists(db_path):
        logger.warning(f"Plex database not found at: {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query for all media parts associated with this item
        query = """
            SELECT media_parts.file, media_parts.hash
            FROM media_parts
            JOIN media_items ON media_parts.media_item_id = media_items.id
            WHERE media_items.metadata_item_id = ?
        """

        cursor.execute(query, (rating_key,))
        results = cursor.fetchall()

        conn.close()

        if results:
            return [(file_path, bundle_hash) for file_path, bundle_hash in results if file_path]
        else:
            logger.warning(f"No media parts found in database for item {rating_key}")
            return []

    except sqlite3.Error as e:
        logger.error(f"Database error while querying media parts for item {rating_key}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error querying database for item {rating_key}: {type(e).__name__}: {e}")
        return []


def get_all_media_parts_batch(plex_config_folder: str, rating_keys: list[int]) -> dict:
    """
    Query media parts for multiple items at once from the Plex SQLite database.

    This is much more efficient than calling get_media_parts_from_database() for each item individually.

    Args:
        plex_config_folder: Path to Plex config folder
        rating_keys: List of Plex rating keys (item IDs)

    Returns:
        dict: Dictionary mapping rating_key -> [(file_path, bundle_hash), ...]
    """
    if not rating_keys:
        return {}

    db_path = os.path.join(plex_config_folder, 'Plug-in Support', 'Databases',
                           'com.plexapp.plugins.library.db')

    if not os.path.exists(db_path):
        logger.warning(f"Plex database not found at: {db_path}")
        return {}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query for all media parts for all rating keys at once
        placeholders = ','.join('?' * len(rating_keys))
        query = f"""
            SELECT media_items.metadata_item_id, media_parts.file, media_parts.hash
            FROM media_parts
            JOIN media_items ON media_parts.media_item_id = media_items.id
            WHERE media_items.metadata_item_id IN ({placeholders})
        """

        cursor.execute(query, rating_keys)
        results = cursor.fetchall()
        conn.close()

        # Build dictionary mapping rating_key -> list of (file_path, bundle_hash)
        media_parts_map = {}
        for rating_key, file_path, bundle_hash in results:
            if file_path:  # Only include if file path exists
                if rating_key not in media_parts_map:
                    media_parts_map[rating_key] = []
                media_parts_map[rating_key].append((file_path, bundle_hash))

        return media_parts_map

    except sqlite3.Error as e:
        logger.error(f"Database error while batch querying media parts: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error batch querying database: {type(e).__name__}: {e}")
        return {}


def get_hash_from_database(plex_config_folder: str, file_path: str) -> str:
    """
    Query the bundle hash directly from the Plex SQLite database.

    This is a fallback method when the Plex API doesn't return a hash.

    Note: The bundle hash is stored in media_parts.hash, not metadata_items.hash.
    The metadata_items.hash field is often NULL and serves a different purpose.

    Args:
        plex_config_folder: Path to Plex config folder
        file_path: Full path to the media file

    Returns:
        str: Bundle hash if found, None otherwise
    """
    db_path = os.path.join(plex_config_folder, 'Plug-in Support', 'Databases',
                           'com.plexapp.plugins.library.db')

    if not os.path.exists(db_path):
        logger.warning(f"Plex database not found at: {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query media_parts.hash directly (bundle hash is in media_parts, not metadata_items)
        query = """
            SELECT hash
            FROM media_parts
            WHERE file = ?
            LIMIT 1
        """

        cursor.execute(query, (file_path,))
        result = cursor.fetchone()

        conn.close()

        if result and result[0]:
            return result[0]
        else:
            logger.debug(f"No hash found in database for {file_path}")
            return None

    except sqlite3.Error as e:
        logger.error(f"Database error while querying hash for {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error querying database for {file_path}: {type(e).__name__}: {e}")
        return None

def _convert_timestamp_to_datetime(timestamp):
    """Convert a Unix timestamp (integer) to a datetime object.

    Handles None values and already-converted datetime objects gracefully.
    """
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp
    try:
        return datetime.fromtimestamp(timestamp)
    except (TypeError, ValueError, OSError):
        return None


def get_library_sections_from_database(plex_config_folder: str, plex_libraries: list[str] = None,
                                        sort_by: str = None) -> list:
    """
    Query library sections and media items directly from the Plex SQLite database.

    This is a fallback when the Plex API is unavailable (returning 500 errors).
    It queries the database directly to get all movies and episodes.

    Args:
        plex_config_folder: Path to Plex config folder
        plex_libraries: Optional list of library names to filter (lowercase)
        sort_by: Optional sort order ('newest' or 'oldest')

    Returns:
        list: List of tuples (section_title, media_items) where media_items is a list of
              (rating_key, title, media_type, bundle_hash, added_at) tuples
    """
    db_path = os.path.join(plex_config_folder, 'Plug-in Support', 'Databases',
                           'com.plexapp.plugins.library.db')

    if not os.path.exists(db_path):
        logger.error(f"Plex database not found at: {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get library sections
        sections_query = """
            SELECT id, name, section_type
            FROM library_sections
            WHERE section_type IN (1, 2)  -- 1=movie, 2=show
        """
        cursor.execute(sections_query)
        sections = cursor.fetchall()

        results = []

        for section in sections:
            section_id = section['id']
            section_name = section['name']
            section_type = section['section_type']

            # Skip if not in configured libraries
            if plex_libraries and section_name.lower() not in plex_libraries:
                logger.info(f"Skipping library '{section_name}' as it's not in the configured libraries list")
                continue

            logger.info(f"Getting media files from library '{section_name}' (database mode)...")

            if section_type == 1:  # Movies
                # Determine sort order for movies (uses 'mi' alias)
                order_clause = "ORDER BY mi.added_at DESC"  # Default: newest first
                if sort_by == 'oldest':
                    order_clause = "ORDER BY mi.added_at ASC"
                # Query movies with their bundle hashes
                query = f"""
                    SELECT mi.id as rating_key, mi.title, mi.added_at, mp.hash as bundle_hash
                    FROM metadata_items mi
                    JOIN media_items mei ON mei.metadata_item_id = mi.id
                    JOIN media_parts mp ON mp.media_item_id = mei.id
                    WHERE mi.library_section_id = ?
                    AND mi.metadata_type = 1  -- Movie
                    GROUP BY mi.id
                    {order_clause}
                """
                cursor.execute(query, (section_id,))
                rows = cursor.fetchall()

                media_items = [
                    (row['rating_key'], row['title'], 'movie', row['bundle_hash'],
                     _convert_timestamp_to_datetime(row['added_at']))
                    for row in rows
                ]

            elif section_type == 2:  # TV Shows
                # Determine sort order for episodes (uses 'ep' alias)
                order_clause = "ORDER BY ep.added_at DESC"  # Default: newest first
                if sort_by == 'oldest':
                    order_clause = "ORDER BY ep.added_at ASC"
                # Query episodes with show title formatted as "Show S01E01"
                # Note: season number is stored in season.index, not ep.parent_index
                query = f"""
                    SELECT
                        ep.id as rating_key,
                        show.title as show_title,
                        season.'index' as season_num,
                        ep.'index' as episode_num,
                        ep.added_at,
                        mp.hash as bundle_hash,
                        mp.file as file_path
                    FROM metadata_items ep
                    JOIN metadata_items season ON ep.parent_id = season.id
                    JOIN metadata_items show ON season.parent_id = show.id
                    JOIN media_items mei ON mei.metadata_item_id = ep.id
                    JOIN media_parts mp ON mp.media_item_id = mei.id
                    WHERE ep.library_section_id = ?
                    AND ep.metadata_type = 4  -- Episode
                    GROUP BY ep.id  -- One row per episode (multiple files handled via media_parts_info)
                    {order_clause}
                """
                cursor.execute(query, (section_id,))
                rows = cursor.fetchall()

                media_items = []
                for row in rows:
                    # Format as "Show Title S01E01"
                    season_num = row['season_num'] or 0
                    episode_num = row['episode_num'] or 0
                    formatted_title = f"{row['show_title']} S{season_num:02d}E{episode_num:02d}"
                    media_items.append(
                        (row['rating_key'], formatted_title, 'episode', row['bundle_hash'],
                         _convert_timestamp_to_datetime(row['added_at']))
                    )
            else:
                continue

            logger.info(f"Retrieved {len(media_items)} media files from library '{section_name}' (database mode)")
            results.append((section_name, media_items))

        conn.close()
        return results

    except sqlite3.Error as e:
        logger.error(f"Database error while querying library sections: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error querying library sections: {type(e).__name__}: {e}")
        return []


def get_watch_history_from_database(plex_config_folder: str, limit_per_user: int = 50) -> dict:
    """
    Query watch history for all users directly from the Plex SQLite database.

    This is more reliable than using the PlexAPI history() method which may only
    return history for the authenticated user.

    Args:
        plex_config_folder: Path to Plex config folder
        limit_per_user: Maximum number of recent items to return per user (default: 50)

    Returns:
        dict: Dictionary mapping account_id -> list of (metadata_item_id, viewed_at) tuples,
              sorted by most recent first
    """
    db_path = os.path.join(plex_config_folder, 'Plug-in Support', 'Databases',
                           'com.plexapp.plugins.library.db')

    if not os.path.exists(db_path):
        logger.warning(f"Plex database not found at: {db_path}")
        return {}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query watch history from metadata_item_views table
        # Join with metadata_items to get the rating key (id)
        # metadata_item_views stores the guid, we need to join to get the actual rating key
        query = """
            SELECT miv.account_id, mi.id as rating_key, miv.viewed_at
            FROM metadata_item_views miv
            JOIN metadata_items mi ON miv.guid = mi.guid
            WHERE miv.account_id IS NOT NULL
            ORDER BY miv.account_id, miv.viewed_at DESC
        """

        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        # Organize by account_id and limit per user
        history_by_user = {}
        for account_id, rating_key, viewed_at in results:
            if account_id not in history_by_user:
                history_by_user[account_id] = []

            # Only add if we haven't reached the limit for this user
            if len(history_by_user[account_id]) < limit_per_user:
                history_by_user[account_id].append((rating_key, viewed_at))

        # Log statistics about the retrieved history
        total_views = sum(len(views) for views in history_by_user.values())
        logger.info(f"Retrieved watch history for {len(history_by_user)} users from database ({total_views} total views)")

        # Log per-user statistics at debug level
        for account_id, views in sorted(history_by_user.items(), key=lambda x: len(x[1]), reverse=True):
            logger.debug(f"  Account {account_id}: {len(views)} views")

        return history_by_user

    except sqlite3.Error as e:
        logger.error(f"Database error while querying watch history: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error querying watch history: {type(e).__name__}: {e}")
        return {}


def get_library_sections(plex, config: Config, database_only: bool = False):
    """
    Get all library sections from Plex server.

    Automatically falls back to querying the SQLite database directly if the
    Plex API returns errors (e.g., 500 errors).

    Args:
        plex: Plex server instance
        config: Configuration object
        database_only: If True, skip API and query database directly (faster)

    Yields:
        tuple: (section, media_items) for each library
    """
    import time

    # Create a simple section-like object for compatibility
    class DatabaseSection:
        def __init__(self, title):
            self.title = title

    # If database_only mode is enabled, skip API entirely
    if database_only:
        logger.info("Using database-only mode for library sync (faster)...")
        db_results = get_library_sections_from_database(
            config.plex_config_folder,
            config.plex_libraries,
            config.sort_by
        )
        if db_results:
            for section_name, media_items in db_results:
                yield DatabaseSection(section_name), media_items
        else:
            logger.error("Database query failed. Cannot proceed without library access.")
        return

    # Step 1: Get all library sections (1 API call)
    logger.info("Getting all Plex library sections...")
    start_time = time.time()

    try:
        sections = retry_plex_call(plex.library.sections)
    except (requests.exceptions.RequestException, http.client.BadStatusLine, xml.etree.ElementTree.ParseError) as e:
        logger.warning(f"Failed to get Plex library sections via API: {e}")
        logger.info("Falling back to database-only mode...")

        # Fall back to database method
        db_results = get_library_sections_from_database(
            config.plex_config_folder,
            config.plex_libraries,
            config.sort_by
        )
        if db_results:
            for section_name, media_items in db_results:
                yield DatabaseSection(section_name), media_items
        else:
            logger.error("Database fallback also failed. Cannot proceed without library access.")
        return

    sections_time = time.time() - start_time
    logger.info(f"Retrieved {len(sections)} library sections in {sections_time:.2f} seconds")

    # Step 2: Filter and process each library
    for section in sections:
        # Skip libraries that aren't in the PLEX_LIBRARIES list if it's not empty
        if config.plex_libraries and section.title.lower() not in config.plex_libraries:
            logger.info('Skipping library \'{}\' as it\'s not in the configured libraries list'.format(section.title))
            continue

        logger.info('Getting media files from library \'{}\'...'.format(section.title))
        library_start_time = time.time()

        # Determine sort parameter if sort_by is configured
        sort_param = None
        if config.sort_by:
            if config.sort_by == 'newest':
                sort_param = 'addedAt:desc'
            elif config.sort_by == 'oldest':
                sort_param = 'addedAt:asc'

        try:
            if section.METADATA_TYPE == 'episode':
                # Get episodes with locations for duplicate filtering
                search_kwargs = {'libtype': 'episode'}
                if sort_param:
                    search_kwargs['sort'] = sort_param
                search_results = retry_plex_call(section.search, **search_kwargs)
                media_with_locations = []
                for m in search_results:
                    # Format episode title as "Show Title S01E01"
                    show_title = m.grandparentTitle
                    season_episode = m.seasonEpisode.upper()
                    formatted_title = f"{show_title} {season_episode}"
                    media_with_locations.append((m.ratingKey, m.locations, formatted_title, 'episode', get_hash_with_fallback(m, config.plex_config_folder), m.addedAt))
                # Filter out multi episode files based on file locations
                media = filter_duplicate_locations(media_with_locations)
            elif section.METADATA_TYPE == 'movie':
                search_kwargs = {}
                if sort_param:
                    search_kwargs['sort'] = sort_param
                search_results = retry_plex_call(section.search, **search_kwargs)
                media = [(m.ratingKey, m.title, 'movie', get_hash_with_fallback(m, config.plex_config_folder), m.addedAt) for m in search_results]
            else:
                logger.info('Skipping library {} as \'{}\' is unsupported'.format(section.title, section.METADATA_TYPE))
                continue
        except (requests.exceptions.RequestException, http.client.BadStatusLine, xml.etree.ElementTree.ParseError) as e:
            logger.warning(f"Failed to search library '{section.title}' via API: {e}")
            logger.info(f"Falling back to database for library '{section.title}'...")

            # Fall back to database for this specific library
            db_results = get_library_sections_from_database(
                config.plex_config_folder,
                [section.title.lower()],  # Only query this library
                config.sort_by
            )
            if db_results:
                for section_name, media_items in db_results:
                    library_time = time.time() - library_start_time
                    logger.info(f"Retrieved {len(media_items)} media files from library '{section_name}' via database in {library_time:.2f} seconds")
                    yield section, media_items
            else:
                logger.error(f"Database fallback failed for library '{section.title}'. Skipping.")
            continue

        library_time = time.time() - library_start_time
        logger.info('Retrieved {} media files from library \'{}\' in {:.2f} seconds'.format(len(media), section.title, library_time))
        yield section, media
