#!/usr/bin/env python3
"""
Restore RGB semantic symlinks for Ctrl-V evaluation.

This reverses the fix_semantic_symlinks.py script, converting ID-based semantic
symlinks back to RGB semantic_rgb symlinks (required for Ctrl-V model which
was trained on RGB semantic maps).
"""

import os
from pathlib import Path
import argparse

def restore_symlinks(semantics_dir, dry_run=False):
    """
    Restore symlinks from ID-based (semantic) to RGB (semantic_rgb).
    
    Args:
        semantics_dir: Path to semantics directory
        dry_run: If True, only print what would be done
    """
    print("=" * 80)
    print("Restoring RGB Semantic Symlinks for Ctrl-V")
    print("=" * 80)
    print(f"\nProcessing directory: {semantics_dir}")
    print(f"Dry run: {dry_run}\n")
    
    if not os.path.exists(semantics_dir):
        print(f"❌ ERROR: Directory not found: {semantics_dir}")
        return 1
    
    fixed_count = 0
    error_count = 0
    skipped_count = 0
    
    # Walk through all scene directories
    for scene_dir in sorted(os.listdir(semantics_dir)):
        scene_path = os.path.join(semantics_dir, scene_dir)
        
        if not os.path.isdir(scene_path):
            continue
        
        print(f"Processing scene: {scene_dir}")
        
        # Process each file in the scene directory
        for filename in sorted(os.listdir(scene_path)):
            file_path = os.path.join(scene_path, filename)
            
            # Check if it's a symlink
            if not os.path.islink(file_path):
                if dry_run:
                    print(f"  Skipping (not a symlink): {filename}")
                skipped_count += 1
                continue
            
            # Get current target
            current_target = os.readlink(file_path)
            
            # Check if it points to ID-based semantic (not semantic_rgb)
            if '/semantic/' in current_target and '/semantic_rgb/' not in current_target:
                # Replace semantic with semantic_rgb
                new_target = current_target.replace('/semantic/', '/semantic_rgb/')
                
                # Verify new target exists
                if os.path.exists(new_target):
                    if dry_run:
                        print(f"  Would update: {filename}")
                        print(f"    Old: {current_target}")
                        print(f"    New: {new_target}")
                    else:
                        # Remove old symlink and create new one
                        os.remove(file_path)
                        os.symlink(new_target, file_path)
                        print(f"  ✓ Updated: {filename}")
                    fixed_count += 1
                else:
                    print(f"  ⚠️  Warning: RGB target doesn't exist: {new_target}")
                    print(f"      Current target: {current_target}")
                    error_count += 1
            elif '/semantic_rgb/' in current_target:
                if dry_run:
                    print(f"  Already RGB: {filename}")
                skipped_count += 1
            else:
                if dry_run:
                    print(f"  Unknown format: {filename} -> {current_target}")
                skipped_count += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Updated symlinks:  {fixed_count}")
    print(f"⏭️  Skipped (already RGB or not symlink): {skipped_count}")
    print(f"⚠️  Errors:           {error_count}")
    print("=" * 80)
    
    if error_count > 0:
        print("\n⚠️  Some symlinks could not be updated. Check warnings above.")
        print("   This likely means the RGB semantic files don't exist in KITTI-360.")
        return 1
    
    if dry_run:
        print("\n🔍 This was a DRY RUN - no files were actually changed.")
        print("   Remove --dry_run to apply changes.")
    else:
        print("\n✅ All symlinks successfully restored to RGB semantic_rgb!")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Restore RGB semantic symlinks for Ctrl-V evaluation'
    )
    parser.add_argument(
        '--semantics_dir',
        type=str,
        default='/no_backups/s1492/kitti360_ctrlv/semantics/track/val',
        help='Path to semantics directory (default: validation set)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print what would be done without actually doing it'
    )
    args = parser.parse_args()
    
    return restore_symlinks(args.semantics_dir, args.dry_run)


if __name__ == "__main__":
    exit(main())
