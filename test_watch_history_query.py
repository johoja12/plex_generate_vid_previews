#!/usr/bin/env python3
"""
Test script to verify the watch history database query works correctly.
Run this on your Plex server to test the query.
"""

import sqlite3
import sys
import os

def test_watch_history_query(plex_config_folder: str):
    """Test the watch history query against the Plex database."""

    db_path = os.path.join(plex_config_folder, 'Plug-in Support', 'Databases',
                           'com.plexapp.plugins.library.db')

    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return False

    print(f"✓ Found database at: {db_path}")
    print()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # First, check what columns exist in metadata_item_views
        print("=== Checking metadata_item_views table schema ===")
        cursor.execute("PRAGMA table_info(metadata_item_views)")
        columns = cursor.fetchall()
        print("Columns in metadata_item_views:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        print()

        # Check metadata_items columns too
        print("=== Checking metadata_items table schema ===")
        cursor.execute("PRAGMA table_info(metadata_items)")
        columns = cursor.fetchall()
        print("Key columns in metadata_items:")
        for col in columns[:10]:  # Show first 10
            print(f"  - {col[1]} ({col[2]})")
        print()

        # Check how many unique accounts have watch history
        print("=== Checking total users with watch history ===")
        cursor.execute("""
            SELECT COUNT(DISTINCT account_id) as user_count
            FROM metadata_item_views
            WHERE account_id IS NOT NULL
        """)
        user_count = cursor.fetchone()[0]
        print(f"Total unique users with watch history: {user_count}")
        print()

        # Check total view records
        cursor.execute("SELECT COUNT(*) FROM metadata_item_views WHERE account_id IS NOT NULL")
        total_views = cursor.fetchone()[0]
        print(f"Total view records: {total_views}")
        print()

        # Check if there are view records that don't match in metadata_items
        cursor.execute("""
            SELECT COUNT(*)
            FROM metadata_item_views miv
            LEFT JOIN metadata_items mi ON miv.guid = mi.guid
            WHERE miv.account_id IS NOT NULL
            AND mi.id IS NULL
        """)
        unmatched = cursor.fetchone()[0]
        if unmatched > 0:
            print(f"⚠️  Warning: {unmatched} view records don't have matching metadata_items")
            print(f"   This could be deleted items. These will be excluded from results.")
        else:
            print(f"✓ All view records have matching metadata_items")
        print()

        # Test the actual query
        print("=== Testing watch history query ===")
        query = """
            SELECT miv.account_id, mi.id as rating_key, miv.viewed_at
            FROM metadata_item_views miv
            JOIN metadata_items mi ON miv.guid = mi.guid
            WHERE miv.account_id IS NOT NULL
            ORDER BY miv.account_id, miv.viewed_at DESC
            LIMIT 20
        """

        print("Query:")
        print(query)
        print()

        cursor.execute(query)
        results = cursor.fetchall()

        print(f"✓ Query executed successfully!")
        print(f"✓ Retrieved {len(results)} rows (limited to 20 for testing)")
        print()

        if results:
            print("Sample results:")
            print(f"{'Account ID':<15} {'Rating Key':<15} {'Viewed At':<25}")
            print("-" * 60)
            for account_id, rating_key, viewed_at in results[:10]:
                print(f"{account_id:<15} {rating_key:<15} {viewed_at:<25}")

            # Count unique users
            unique_accounts = set(r[0] for r in results)
            print()
            print(f"✓ Found history for {len(unique_accounts)} unique account(s) in sample")

            # Organize by account
            history_by_user = {}
            for account_id, rating_key, viewed_at in results:
                if account_id not in history_by_user:
                    history_by_user[account_id] = []
                history_by_user[account_id].append((rating_key, viewed_at))

            print()
            print("Items per account (in sample):")
            for account_id, items in history_by_user.items():
                print(f"  Account {account_id}: {len(items)} items")

            # Now get FULL count per user (not limited to 20)
            print()
            print("=== Full user statistics (all history) ===")
            cursor.execute("""
                SELECT miv.account_id, COUNT(*) as view_count
                FROM metadata_item_views miv
                JOIN metadata_items mi ON miv.guid = mi.guid
                WHERE miv.account_id IS NOT NULL
                GROUP BY miv.account_id
                ORDER BY view_count DESC
            """)
            full_stats = cursor.fetchall()
            print(f"✓ Total users in full query: {len(full_stats)}")
            print()
            print("Complete breakdown by account_id:")
            for account_id, count in full_stats:
                print(f"  Account {account_id}: {count} total views")
        else:
            print("⚠️  No results returned - this might mean:")
            print("   1. No watch history exists in database")
            print("   2. The join condition is incorrect")
            print("   3. All account_id values are NULL")

        conn.close()
        return True

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_watch_history_query.py <path_to_plex_config_folder>")
        print()
        print("Example:")
        print("  python test_watch_history_query.py '/var/lib/plexmediaserver/Library/Application Support/Plex Media Server'")
        print("  python test_watch_history_query.py '/config/Library/Application Support/Plex Media Server'")
        sys.exit(1)

    plex_config_folder = sys.argv[1]
    success = test_watch_history_query(plex_config_folder)

    sys.exit(0 if success else 1)
