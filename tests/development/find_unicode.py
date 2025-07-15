#!/usr/bin/env python3
"""
Find all non-ASCII Unicode characters in Python source files
"""
import os
import sys

def find_unicode_in_file(filepath):
    """Find Unicode characters in a file"""
    unicode_chars = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                for col_num, char in enumerate(line):
                    if ord(char) > 127:  # Non-ASCII
                        unicode_chars.append({
                            'line': line_num,
                            'col': col_num,
                            'char': char,
                            'code': f'U+{ord(char):04X}',
                            'context': line.strip()
                        })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return unicode_chars

def main():
    """Main function"""
    src_dir = "src"
    found_unicode = False
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                unicode_chars = find_unicode_in_file(filepath)
                
                if unicode_chars:
                    found_unicode = True
                    print(f"\n{filepath}:")
                    for char_info in unicode_chars:
                        print(f"  Line {char_info['line']}, Col {char_info['col']}: '{char_info['char']}' ({char_info['code']})")
                        print(f"    Context: {char_info['context']}")
    
    if not found_unicode:
        print("No Unicode characters found in Python source files.")
    else:
        print("\nNote: Replace these Unicode characters with ASCII equivalents for Pi compatibility.")

if __name__ == "__main__":
    main()