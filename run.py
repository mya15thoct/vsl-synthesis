#!/usr/bin/env python3
"""
VSL Synthesis - Main Runner Script

Simple script to run synthesis on server with command-line arguments.

Usage:
    python run.py hello my name --method spline
    python run.py hello world --method bezier --output test.mp4
    python run.py --list  # List available words
"""

import argparse
import sys
from pathlib import Path

# Import configuration and pipeline
from config import DATA_DIR, OUTPUT_DIR, IS_SERVER, FPS, TRANSITION_FRAMES
from src.pipeline import synthesize_sentence


def list_available_words():
    """List all available .npy files in DATA_DIR."""
    print(f"\nAvailable words in: {DATA_DIR}")
    print("="*60)
    
    if not DATA_DIR.exists():
        print(f"Directory not found: {DATA_DIR}")
        return
    
    # Look for folders containing .npy files
    word_folders = []
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            # Check if folder contains .npy files
            npy_files = list(item.glob("*.npy"))
            if npy_files:
                word_folders.append(item.name)
    
    # Also check for direct .npy files
    direct_npy = [f.stem for f in DATA_DIR.glob("*.npy")]
    
    all_words = sorted(word_folders + direct_npy)
    
    if not all_words:
        print("No words found!")
        print("Expected structure:")
        print("  sequences/Word1/*.npy  OR  sequences/word1.npy")
        return
    
    print(f"Found {len(all_words)} words:\n")
    
    # Display in 4 columns
    for i in range(0, len(all_words), 4):
        row = all_words[i:i+4]
        print("  ".join(f"{w:15s}" for w in row))
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='VSL Synthesis - Generate sign language videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py hello my name
  python run.py hello world --method bezier
  python run.py hello my name is Ram --output sentence.mp4
  python run.py --list
        """
    )
    
    parser.add_argument('words', nargs='*', help='Words to synthesize (e.g., hello my name)')
    parser.add_argument('--method', '-m', default='spline', 
                       choices=['linear', 'spline', 'bezier'],
                       help='Interpolation method (default: spline)')
    parser.add_argument('--output', '-o', help='Output filename (default: auto-generated)')
    parser.add_argument('--list', '-l', action='store_true', 
                       help='List available words')
    parser.add_argument('--fps', type=int, default=FPS, 
                       help=f'Frames per second (default: {FPS})')
    parser.add_argument('--transition-frames', type=int, default=TRANSITION_FRAMES,
                       help=f'Transition frames (default: {TRANSITION_FRAMES})')
    
    args = parser.parse_args()
    
    # Print environment info
    print(f"\n VSL Synthesis")
    print("="*60)
    print(f"Environment: {'SERVER' if IS_SERVER else 'LOCAL'}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # List available words
    if args.list:
        list_available_words()
        return
    
    # Check if words provided
    if not args.words:
        print("\nNo words provided!")
        print("Usage: python run.py hello my name")
        print("Or run: python run.py --list")
        sys.exit(1)
    
    # Find .npy files for each word
    print(f"\n Looking for {len(args.words)} words...")
    word_files = []
    missing_words = []
    
    for word in args.words:
        # Try folder first (e.g., sequences/Afternoon/)
        # Case-insensitive search: try original, Title case, and UPPER case
        word_variants = [word, word.title(), word.upper(), word.lower()]
        
        found = False
        word_folder = None
        word_file = None
        
        # Find matching folder or file (case-insensitive)
        for variant in word_variants:
            test_folder = DATA_DIR / variant
            test_file = DATA_DIR / f"{variant}.npy"
            if test_folder.is_dir():
                word_folder = test_folder
                break
            if test_file.exists():
                word_file = test_file
                break
        
        # Case 1: Folder with .npy files
        if word_folder and word_folder.is_dir():
            npy_files = sorted(word_folder.glob("*.npy"))
            if npy_files:
                word_files.append(str(word_folder))  # Pass folder path, will be resolved later
                print(f"  {word:15s} -> {word_folder.name}/ (using {npy_files[0].name})")
                found = True
        
        # Case 2: Direct .npy file
        if not found and word_file and word_file.exists():
            word_files.append(str(word_file))
            print(f"  {word:15s} -> {word_file.name}")
            found = True
        
        if not found:
            missing_words.append(word)
            print(f"   {word:15s} -> NOT FOUND")
    
    if missing_words:
        print(f"\nMissing {len(missing_words)} words: {', '.join(missing_words)}")
        print("Run with --list to see available words")
        
        if len(word_files) == 0:
            print("No valid words found. Exiting.")
            sys.exit(1)
        
        print(f"\nContinuing with {len(word_files)} valid words...")
    
    # Generate output filename
    if args.output:
        output_name = args.output
    else:
        sentence_str = "_".join(args.words[:5])
        if len(args.words) > 5:
            sentence_str += "_etc"
        output_name = f"{sentence_str}_{args.method}.mp4"
    
    output_path = OUTPUT_DIR / "baseline" / output_name
    
    # Run synthesis
    print(f"\n🎥 Synthesizing with {args.method} method...")
    print(f"Output: {output_path}")
    
    try:
        result = synthesize_sentence(
            word_videos=word_files,
            method=args.method,
            output_path=str(output_path),
            transition_frames=args.transition_frames,
            fps=args.fps
        )
        
        print(f"\nSUCCESS!")
        print(f"Video saved to: {result}")
        
    except Exception as e:
        print(f"\nError during synthesis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
