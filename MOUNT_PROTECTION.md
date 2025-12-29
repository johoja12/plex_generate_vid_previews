# Mount Protection Guide

## Overview

Mount Protection is a critical feature that prevents the application from incorrectly marking all your media as missing when mount points (like rclone, NFS, SMB) are temporarily unavailable. This commonly happens when:

- rclone mount crashes or restarts
- NAS goes offline temporarily
- Network issues cause mount failures
- Docker container starts before mounts are ready

Without this protection, the sync process could incorrectly mark thousands of items as `MEDIA_MISSING`, causing data corruption in your database.

## How It Works

Mount Protection uses three layers of defense:

### 1. **Path Validation** (Proactive)
Before syncing or processing, the system checks if configured marker paths exist and are accessible. If validation fails:
- Sync is aborted immediately
- Queue is auto-paused
- Clear error messages are logged
- Changes are NOT committed to database

### 2. **Threshold Detection** (Reactive)
During sync, if more than X% of items would suddenly become MEDIA_MISSING (default: 50%):
- Transaction is rolled back
- No changes are written to database
- Queue is auto-paused
- Administrator is alerted via logs

### 3. **Processing Protection**
Before each queue processing cycle:
- Mount validation is performed
- If failed, processing is skipped
- Queue is auto-paused automatically

## Configuration

### Web Interface

1. Navigate to **Settings** → **Mount Protection**
2. Configure the following:

#### Enable Mount Validation
- Toggle: Enable/disable mount checking
- Default: **Enabled** (recommended)

#### Mount Check Paths
- Comma-separated list of paths to validate
- Example: `/mnt/media/.mounted,/data/movies/.marker,/mnt/nas/.ready`
- **Recommended**: Create marker files in your mount directories

#### Failure Threshold (%)
- Percentage of items that can become missing before aborting
- Default: **50%**
- Range: 1-100%
- Lower = more sensitive, Higher = more tolerant

### Creating Marker Files

The recommended approach is to create marker files in your mount directories:

```bash
# On your host (where rclone/mount is active)
# Create marker files in each mount point
touch /mnt/media/.mounted
touch /data/movies/.marker

# Make them read-only to prevent accidental deletion
chmod 444 /mnt/media/.mounted
chmod 444 /data/movies/.marker
```

Then configure these paths in the settings:
```
/mnt/media/.mounted,/data/movies/.marker
```

### Docker Example

If using Docker with volume mounts:

```yaml
# docker-compose.yml
services:
  plex-previews:
    image: plex-generate-previews
    volumes:
      - /mnt/media:/media
      - /data/movies:/movies
    environment:
      - MOUNT_CHECK_PATHS=/media/.mounted,/movies/.marker
```

Create marker files on the host:
```bash
touch /mnt/media/.mounted
touch /data/movies/.marker
```

### rclone Integration

If using rclone mount, you can use a systemd service to create/remove marker files:

```ini
# /etc/systemd/system/rclone-media.service
[Unit]
Description=RClone Media Mount
After=network-online.target

[Service]
Type=notify
ExecStartPre=/bin/mkdir -p /mnt/media
ExecStart=/usr/bin/rclone mount remote:media /mnt/media --vfs-cache-mode writes
ExecStartPost=/bin/touch /mnt/media/.mounted
ExecStop=/bin/rm -f /mnt/media/.mounted
ExecStop=/bin/fusermount -u /mnt/media
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

This ensures the marker file only exists when the mount is active.

## Testing Mount Protection

### Test Button
In the Settings → Mount Protection section, click **"Test Mount Validation"** to:
- Verify configured paths are accessible
- See which paths are missing (if any)
- Confirm your configuration is correct

### Manual Testing

1. **Test mount failure detection:**
   ```bash
   # Unmount your media
   fusermount -u /mnt/media

   # Trigger a sync via web UI
   # Check logs - should see: "MOUNT VALIDATION FAILED"
   # Queue should auto-pause

   # Remount media
   rclone mount remote:media /mnt/media

   # Resume queue via web UI
   ```

2. **Test threshold detection:**
   ```bash
   # Temporarily move media files
   mv /mnt/media /mnt/media.backup

   # Trigger sync
   # Should see: "SANITY CHECK FAILED"
   # Changes should be rolled back

   # Restore media
   mv /mnt/media.backup /mnt/media
   ```

## Log Messages

### Success Messages
```
Mount validation disabled, skipping check
No mount check paths configured, validation passes
Mount check passed: /mnt/media/.mounted
```

### Warning Messages
```
Mount check failed: Path does not exist: /mnt/media/.mounted
Mount validation failed during queue processing
Skipping queue processing until mounts are available
```

### Error Messages
```
⚠️⚠️⚠️  MOUNT VALIDATION FAILED - Sync aborted! ⚠️⚠️⚠️
Missing/inaccessible paths: /mnt/media/.mounted, /data/.marker
Auto-pausing queue due to mount failure

⚠️⚠️⚠️  SANITY CHECK FAILED - Sync aborted! ⚠️⚠️⚠️
500/1000 items (50.0%) would be marked as MEDIA_MISSING
This exceeds the threshold of 50% and likely indicates a mount failure
Changes rolled back. Please check your media mounts and retry sync
```

## Troubleshooting

### Mount validation always fails
- Verify paths are correct (case-sensitive on Linux)
- Check Docker volume mounts match configured paths
- Ensure marker files exist with correct permissions
- Test with: `docker exec -it <container> ls -la /mnt/media/.mounted`

### False positives (mount is OK but validation fails)
- Check file permissions (container user must have read access)
- Verify SELinux/AppArmor isn't blocking access
- Try directory paths instead of files (e.g., `/mnt/media` instead of `/mnt/media/.mounted`)

### Sync keeps getting paused
- Check mount stability (is rclone crashing?)
- Review threshold setting (might be too low)
- Check logs for specific paths failing
- Verify network connectivity to NAS/cloud storage

### I want to disable protection temporarily
1. Go to Settings → Mount Protection
2. Uncheck "Enable Mount Validation"
3. Save settings
4. Resume queue if paused

**Warning**: Only disable if you're certain mounts are stable!

## Best Practices

1. **Always use marker files**: More reliable than checking mount points directly
2. **Test your configuration**: Use the Test button before going live
3. **Monitor logs**: Watch for mount validation warnings
4. **Adjust threshold**: Lower for critical systems, higher for unstable mounts
5. **Automate mount health**: Use systemd, docker healthchecks, or watchdogs
6. **Document your setup**: Note which paths correspond to which mounts

## FAQ

**Q: What happens if I disable mount protection?**
A: The system will sync and process without checking mounts. If mounts are down, it will mark all items as MEDIA_MISSING.

**Q: Can I use directories instead of marker files?**
A: Yes, but marker files are more reliable. Empty mount points may still be directories.

**Q: Does this slow down processing?**
A: No, validation takes milliseconds and only runs before sync/processing, not per-item.

**Q: What if my mount comes back while paused?**
A: Simply click "Resume Queue" in the web UI. Mount validation will pass and processing resumes.

**Q: Can I have multiple mount check paths?**
A: Yes, comma-separate them. ALL paths must be accessible for validation to pass.

**Q: Does this work with NFS/SMB/other mounts?**
A: Yes, it works with any filesystem mount. Create marker files in the mounted directories.
