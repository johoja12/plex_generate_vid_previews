# Library sync performance

Routine library syncs cache successful media-target existence checks for 24 hours.
Each reuse still checks the library entry's inode, size, modification time and
change time with `lstat`; changed entries invalidate the cached result. New items,
changed bundle hashes or part inventories, and `MEDIA_MISSING` items always check
the target. Failed checks are never cached. BIF existence and the complete Plex
catalogue are still reconciled on every sync, including orphan and mount guards.

A remote target deleted behind an unchanged symlink can remain listed as available
until the cache expires. Workers independently check media before processing.
The cache is in memory, so the first sync after a service restart repopulates it.
Set `MEDIA_AVAILABILITY_CACHE_SECONDS=0` to restore full target checks every sync,
or set another duration in seconds (default `86400`).

InfiniDysk verified-cache details use at most eight concurrent requests, each with
its own HTTP session. Eligibility still requires online storage, complete verified
bytes, matching current size and valid library mappings. A failed request discards
the partial catalogue.

Docker builds run `scripts/verify_runtime_package.py` after installation. It
resolves imports as the installed web console script does and compares installed
Python modules with the source tree, excluding the generated version module.
For a selective image update, update the installed package as well as `/app` and
run this check; a successful import from `/app` does not verify the web runtime.
