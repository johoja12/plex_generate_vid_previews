# Changelog

All notable changes to this fork will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Broken Symlink Detection**: Automatically detects and marks media files with broken symlinks or deleted files as MEDIA_MISSING
  - Checks during startup cleanup for stale PROCESSING/QUEUED items
  - Checks during Plex library sync for new and existing items
  - New MEDIA_MISSING status with purple badge in web UI
  - Prevents wasted worker time on non-existent files
- **Stale Item Cleanup on Startup**: Automatically resets items stuck in PROCESSING/QUEUED status when the application starts
  - Fixes issues after container restarts or worker crashes
  - Checks if media files exist before resetting to MISSING
  - Logs cleanup statistics (items reset, media missing count)
- **Failed Counter**: Dashboard now displays count of failed items (FAILED + SLOW_FAILED + MEDIA_MISSING)
- **Database Files to .gitignore**: Runtime database files are now excluded from version control

### Changed
- **Queue Priority Logic**: Changed from `added_at` to `updated_at` sorting to prioritize recently updated items
  - Items with newly added parts now get queued promptly
  - Timestamp is updated when items are reset to MISSING
- **Startup Initialization**: Stale item cleanup now runs automatically after worker pool initialization
- **Stats API**: Added `media_missing` count to `/api/stats` endpoint

### Fixed
- **Duplicate Queueing**: Fixed items being queued repeatedly by only fetching MISSING items (not QUEUED items)
  - Eliminates "Skipping duplicate assignment" warnings for multi-part media
- **Multi-Select Checkbox**: Fixed select-all checkbox to work correctly with per-part rows
  - Uses unique `item_id` values and deduplicates with Set
- **Expandable Details**: Restored expand/collapse functionality for per-part rows
  - Each part row can be expanded independently
  - Shows detailed information (paths, hashes, errors) for specific parts

## [2024-12-17] - Multi-Part Media Support

### Added
- **Per-Part Row Display**: Media items with multiple versions (e.g., 1080p, 4K) now show as separate table rows
  - Each part displays its own status, progress, and processing speed
  - Easier to see which specific files are completed vs. missing
  - Part index column shows which version (Part 1, Part 2, etc.)
- **Per-Part Status Tracking**: Individual status determination for each media part
  - Parts with existing BIF files always show as COMPLETED
  - Only parts missing BIF files inherit parent's status (MISSING, QUEUED, PROCESSING)
- **Expandable Row Details**: Click ▶ arrow to view detailed information per part
  - Item ID and part index
  - Bundle hash
  - Media file path for this specific part
  - BIF file path for this specific part
  - Error messages (if failed)
- **Toast Notifications**: Replaced blocking JavaScript dialogs with non-blocking toast notifications
  - Better user experience
  - Actions complete without requiring manual dismissal

### Changed
- **Media Items API**: `/api/items` endpoint now expands `media_parts_info` into separate rows
  - Each part gets unique ID: `{item_id}_part{index}`
  - BIF file existence is checked per-part from filesystem
- **Table Structure**: Uses valid HTML with `<tbody>` wrappers instead of React fragments
  - Fixes browser rendering issues with expandable rows
- **Item Selection**: Multi-select now works with unique `item_id` values across all parts

### Fixed
- **Completed Parts Stay Completed**: During processing, parts that already have BIF files maintain COMPLETED status
  - Only missing parts are generated when an item has multiple versions
  - Prevents redundant re-processing of existing previews

## [2024-12-16] - Queue Management Improvements

### Added
- **Average Processing Speed**: Display in web UI and worker logs
  - Shows 5-minute rolling average
  - Helps identify slow-processing items
  - Displayed in worker status logging

### Changed
- **Database-Only Processing**: Eliminated redundant Plex API calls during processing
  - Reads bundle hash directly from Plex database instead of API
  - Significantly reduces API load on Plex server
  - Faster processing with less overhead

### Fixed
- **Plex Connection Error Handling**: Queue auto-pauses on Plex connection errors instead of marking items as failed
  - Prevents mass failures during temporary network issues
  - Resume processing when connection is restored
- **Multi-Part Codec Errors**: Processes all media parts even if one has a codec error
  - Codec errors on one version don't block processing of other versions
  - Each part is processed independently

## [2024-12-15] - Web UI Enhancements

### Added
- **Items Per Page Dropdown**: Configurable pagination in media table
  - Options: 20, 50, 100, 200 items per page
  - Preference saved in browser
- **Frontend Authentication Timeout**: Extended to 1 year
  - Reduces need for frequent re-authentication
  - Better user experience for long-running installations

### Changed
- **Multi-Part Media Tracking**: Track and display multiple media parts/versions in UI
  - JSON storage for `media_parts_info` field
  - Array of objects containing file_path, bundle_hash, bif_path per part
  - Foundation for per-part status tracking

### Fixed
- **New Version Detection**: Detects and requeues completed items with new versions missing BIF files
  - Handles case where 4K version is added to existing 1080p item
  - Automatically resets to MISSING status
  - Updates timestamp for priority queueing

## Earlier Changes

This fork is based on [stevezau/plex_generate_vid_previews](https://github.com/stevezau/plex_generate_vid_previews) with the changes listed above.

For the original project's changelog and features, see the [upstream repository](https://github.com/stevezau/plex_generate_vid_previews).
