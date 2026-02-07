#!/usr/bin/env python3
"""
Update all code references from 554 keypoints to 553 keypoints.

Changes:
- 554 → 553 (keypoint count)
- 1662 → 1659 (flattened feature count: 553 * 3)
- Index ranges in render.py
- Documentation in skeleton_constraints.py
"""

import re
from pathlib import Path

def update_render_py(filepath: Path):
    """Update render.py with correct index ranges."""
    print(f"Updating {filepath}...")
    
    content = filepath.read_text(encoding='utf-8')
    
    # Update keypoint counts
    content = re.sub(r'\b1662\b', '1659', content)
    content = re.sub(r'\b554\b', '553', content)
    
    # Update index ranges for keypoint extraction
    # pose_kpts = skeleton_2d[:44] → [:33]
    content = re.sub(r'skeleton_2d\[:44\]', 'skeleton_2d[:33]', content)
    
    # face_kpts = skeleton_2d[44:512] → [33:511]
    content = re.sub(r'skeleton_2d\[44:512\]', 'skeleton_2d[33:511]', content)
    
    # left_hand_kpts = skeleton_2d[512:533] → [511:532]
    content = re.sub(r'skeleton_2d\[512:533\]', 'skeleton_2d[511:532]', content)
    
    # right_hand_kpts = skeleton_2d[533:554] → [532:553]
    content = re.sub(r'skeleton_2d\[533:554\]', 'skeleton_2d[532:553]', content)
    
    filepath.write_text(content, encoding='utf-8')
    print(f"  ✓ Updated")

def update_skeleton_constraints_py(filepath: Path):
    """Update skeleton_constraints.py with correct structure documentation."""
    print(f"Updating {filepath}...")
    
    content = filepath.read_text(encoding='utf-8')
    
    # Update keypoint counts
    content = re.sub(r'\b1662\b', '1659', content)
    content = re.sub(r'\b554\b', '553', content)
    
    # Update structure comment
    old_comment = """# VSL Skeleton structure (1662 features)
# Pose: 0-131 (33 keypoints × 4: x,y,z,visibility)
# Face: 132-1535 (468 keypoints × 3)
# Left Hand: 1536-1598 (21 keypoints × 3)
# Right Hand: 1599-1661 (21 keypoints × 3)"""
    
    new_comment = """# VSL Skeleton structure (1659 features)
# Pose: 0-98 (33 keypoints × 3: x,y,z)
# Face: 99-1532 (478 keypoints × 3)
# Left Hand: 1533-1595 (21 keypoints × 3)
# Right Hand: 1596-1658 (21 keypoints × 3)"""
    
    content = content.replace(old_comment, new_comment)
    
    # Update index ranges in extract functions
    # Right hand: start, end = 1599, 1662 → 1596, 1659
    content = re.sub(r'start, end = 1599, 1662', 'start, end = 1596, 1659', content)
    
    filepath.write_text(content, encoding='utf-8')
    print(f"  ✓ Updated")

def update_generic_file(filepath: Path):
    """Update generic files with simple replacements."""
    print(f"Updating {filepath}...")
    
    content = filepath.read_text(encoding='utf-8')
    original_content = content
    
    # Replace 1662 → 1659 (must do this first)
    content = re.sub(r'\b1662\b', '1659', content)
    
    # Replace 554 → 553
    content = re.sub(r'\b554\b', '553', content)
    
    if content != original_content:
        filepath.write_text(content, encoding='utf-8')
        print(f"  ✓ Updated")
    else:
        print(f"  - No changes needed")

def main():
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("Updating code for 553 keypoints (MediaPipe Holistic)")
    print("=" * 60)
    print()
    
    # Files needing special handling
    special_files = {
        "src/core/render.py": update_render_py,
        "src/models/skeleton_constraints.py": update_skeleton_constraints_py,
    }
    
    # Files needing generic updates
    generic_files = [
        "src/prepare_data.py",
        "src/train.py",
        "src/models/vsl_diffusion.py",
        "src/models/dataset.py",
        "src/models/mediapipe_loss.py",
        "src/models/custom_scheduler.py",
        "src/methods/vsl_diffusion_adapter.py",
        "src/core/concatenation.py",
    ]
    
    # Update special files
    for file_path, update_func in special_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            update_func(full_path)
        else:
            print(f"  ⚠ File not found: {full_path}")
    
    print()
    
    # Update generic files
    for file_path in generic_files:
        full_path = project_root / file_path
        if full_path.exists():
            update_generic_file(full_path)
        else:
            print(f"  ⚠ File not found: {full_path}")
    
    print()
    print("=" * 60)
    print("✅ Update complete!")
    print("=" * 60)
    print()
    print("Summary of changes:")
    print("  • 554 → 553 keypoints")
    print("  • 1662 → 1659 features (553 × 3)")
    print("  • Updated index ranges in render.py")
    print("  • Updated structure docs in skeleton_constraints.py")
    print()
    print("Next steps:")
    print("  1. Test rendering: python run.py hello How_are_you --method spline")
    print("  2. Prepare training data: python src/prepare_data.py --data_dir /mnt/ngan/vsl_data/sequences")
    print("  3. Retrain model: python src/train.py")

if __name__ == "__main__":
    main()

