#!/usr/bin/env python3
"""
Data Quality Check Script

Kiểm tra chất lượng dữ liệu sau khi extract:
- Format validation
- Shape consistency
- Coordinate range
- Statistics
- Corrupted files detection
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, IS_SERVER


def check_single_file(npy_file: Path):
    """
    Check a single .npy file.
    
    Returns:
        dict with status and info
    """
    try:
        data = np.load(npy_file)
        
        # Basic info
        info = {
            'status': 'ok',
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size_mb': npy_file.stat().st_size / (1024 * 1024),
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std()),
        }
        
        # Validate shape
        if data.ndim != 3:
            info['status'] = 'error'
            info['error'] = f'Expected 3D array, got {data.ndim}D'
            return info
        
        num_frames, num_keypoints, num_coords = data.shape
        
        if num_coords != 3:
            info['status'] = 'error'
            info['error'] = f'Expected 3 coordinates, got {num_coords}'
            return info
        
        if num_keypoints not in [543, 554]:
            info['status'] = 'warning'
            info['warning'] = f'Unexpected keypoint count: {num_keypoints} (expected 543 or 554)'
        
        # Check for NaN or Inf
        if np.isnan(data).any():
            info['status'] = 'error'
            info['error'] = 'Contains NaN values'
            return info
        
        if np.isinf(data).any():
            info['status'] = 'error'
            info['error'] = 'Contains Inf values'
            return info
        
        # Check coordinate range (should be roughly in [-1, 1] or [0, 1])
        if data.max() > 10 or data.min() < -10:
            info['status'] = 'warning'
            info['warning'] = f'Unusual coordinate range: [{data.min():.2f}, {data.max():.2f}]'
        
        # Check for all-zero frames
        zero_frames = np.all(data == 0, axis=(1, 2))
        if zero_frames.any():
            info['status'] = 'warning'
            info['warning'] = f'{zero_frames.sum()} all-zero frames detected'
        
        return info
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def check_data_directory(data_dir: Path, max_files: int = None):
    """
    Check all .npy files in directory.
    """
    print(f"\n{'='*70}")
    print(f"DATA QUALITY CHECK")
    print(f"{'='*70}")
    print(f"Directory: {data_dir}")
    print(f"Server mode: {IS_SERVER}")
    
    # Find all .npy files
    print(f"\n🔍 Scanning for .npy files...")
    
    npy_files = []
    
    # Check if data_dir has word folders
    if data_dir.exists():
        # Look for folders containing .npy files
        for word_folder in data_dir.iterdir():
            if word_folder.is_dir():
                word_npy_files = list(word_folder.glob("*.npy"))
                npy_files.extend(word_npy_files)
        
        # Also check for direct .npy files
        direct_npy = list(data_dir.glob("*.npy"))
        npy_files.extend(direct_npy)
    
    if not npy_files:
        print(f"❌ No .npy files found in {data_dir}")
        return
    
    print(f"✅ Found {len(npy_files)} .npy files")
    
    # Limit for testing
    if max_files:
        npy_files = npy_files[:max_files]
        print(f"   (Checking first {max_files} files for quick test)")
    
    # Check each file
    print(f"\n📊 Checking files...")
    
    results = {
        'ok': [],
        'warning': [],
        'error': []
    }
    
    stats = {
        'shapes': defaultdict(int),
        'keypoint_counts': defaultdict(int),
        'coord_ranges': [],
        'file_sizes': [],
        'num_frames': []
    }
    
    for npy_file in tqdm(npy_files, desc="Checking"):
        info = check_single_file(npy_file)
        
        status = info['status']
        results[status].append({
            'file': str(npy_file.relative_to(data_dir)),
            'info': info
        })
        
        # Collect statistics
        if status in ['ok', 'warning']:
            shape = info['shape']
            stats['shapes'][str(shape)] += 1
            stats['keypoint_counts'][shape[1]] += 1
            stats['coord_ranges'].append((info['min'], info['max']))
            stats['file_sizes'].append(info['size_mb'])
            stats['num_frames'].append(shape[0])
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n📈 Status Distribution:")
    print(f"  ✅ OK:       {len(results['ok']):5d} files ({len(results['ok'])/len(npy_files)*100:.1f}%)")
    print(f"  ⚠️  Warning: {len(results['warning']):5d} files ({len(results['warning'])/len(npy_files)*100:.1f}%)")
    print(f"  ❌ Error:    {len(results['error']):5d} files ({len(results['error'])/len(npy_files)*100:.1f}%)")
    
    # Shape distribution
    print(f"\n📐 Shape Distribution:")
    for shape, count in sorted(stats['shapes'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {shape:30s}: {count:5d} files")
    
    # Keypoint counts
    print(f"\n🎯 Keypoint Counts:")
    for kpt_count, count in sorted(stats['keypoint_counts'].items()):
        print(f"  {kpt_count:3d} keypoints: {count:5d} files")
    
    # Coordinate ranges
    if stats['coord_ranges']:
        coord_mins = [r[0] for r in stats['coord_ranges']]
        coord_maxs = [r[1] for r in stats['coord_ranges']]
        
        print(f"\n📏 Coordinate Ranges:")
        print(f"  Global min: {min(coord_mins):8.4f}")
        print(f"  Global max: {max(coord_maxs):8.4f}")
        print(f"  Avg min:    {np.mean(coord_mins):8.4f}")
        print(f"  Avg max:    {np.mean(coord_maxs):8.4f}")
    
    # Frame statistics
    if stats['num_frames']:
        print(f"\n🎬 Frame Statistics:")
        print(f"  Min frames:  {min(stats['num_frames']):5d}")
        print(f"  Max frames:  {max(stats['num_frames']):5d}")
        print(f"  Avg frames:  {np.mean(stats['num_frames']):5.1f}")
        print(f"  Median:      {np.median(stats['num_frames']):5.1f}")
    
    # File sizes
    if stats['file_sizes']:
        print(f"\n💾 File Size Statistics:")
        print(f"  Min size:  {min(stats['file_sizes']):8.2f} MB")
        print(f"  Max size:  {max(stats['file_sizes']):8.2f} MB")
        print(f"  Avg size:  {np.mean(stats['file_sizes']):8.2f} MB")
        print(f"  Total:     {sum(stats['file_sizes']):8.2f} MB")
    
    # Show errors
    if results['error']:
        print(f"\n{'='*70}")
        print(f"❌ ERRORS ({len(results['error'])} files)")
        print(f"{'='*70}")
        
        for item in results['error'][:20]:  # Show first 20
            print(f"\n  File: {item['file']}")
            print(f"  Error: {item['info'].get('error', 'Unknown error')}")
        
        if len(results['error']) > 20:
            print(f"\n  ... and {len(results['error']) - 20} more errors")
    
    # Show warnings
    if results['warning']:
        print(f"\n{'='*70}")
        print(f"⚠️  WARNINGS ({len(results['warning'])} files)")
        print(f"{'='*70}")
        
        for item in results['warning'][:10]:  # Show first 10
            print(f"\n  File: {item['file']}")
            print(f"  Warning: {item['info'].get('warning', 'Unknown warning')}")
        
        if len(results['warning']) > 10:
            print(f"\n  ... and {len(results['warning']) - 10} more warnings")
    
    # Save detailed report
    report_path = data_dir.parent / 'data_check_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total_files': len(npy_files),
                'ok': len(results['ok']),
                'warning': len(results['warning']),
                'error': len(results['error'])
            },
            'statistics': {
                'shapes': dict(stats['shapes']),
                'keypoint_counts': dict(stats['keypoint_counts']),
                'coord_range': {
                    'min': min(coord_mins) if coord_mins else None,
                    'max': max(coord_maxs) if coord_maxs else None
                },
                'frames': {
                    'min': min(stats['num_frames']) if stats['num_frames'] else None,
                    'max': max(stats['num_frames']) if stats['num_frames'] else None,
                    'avg': float(np.mean(stats['num_frames'])) if stats['num_frames'] else None
                }
            },
            'errors': results['error'][:100],  # Save first 100 errors
            'warnings': results['warning'][:100]  # Save first 100 warnings
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"📄 Detailed report saved to: {report_path}")
    print(f"{'='*70}")
    
    # Final verdict
    print(f"\n{'='*70}")
    if len(results['error']) == 0:
        print(f"✅ DATA CHECK PASSED!")
        if len(results['warning']) > 0:
            print(f"⚠️  {len(results['warning'])} warnings found (review recommended)")
        print(f"{'='*70}")
        return True
    else:
        print(f"❌ DATA CHECK FAILED!")
        print(f"   {len(results['error'])} files have errors")
        print(f"   Please fix errors before training")
        print(f"{'='*70}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check data quality')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory to check (default: from config)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum files to check (for quick test)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Check random sample of N files')
    
    args = parser.parse_args()
    
    # Use config data dir if not specified
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Run check
    success = check_data_directory(data_dir, max_files=args.max_files)
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
