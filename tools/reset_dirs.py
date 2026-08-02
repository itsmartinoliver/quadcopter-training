import os
import shutil
from pathlib import Path

def clear_directory(directory_path):
    """
    Clear all files and subdirectories within a given directory.
    
    Args:
        directory_path (str or Path): Path to the directory to clear
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        path = Path(directory_path)
        
        # Check if directory exists
        if not path.exists():
            print(f"⚠️  Directory does not exist: {directory_path}")
            return False
        
        if not path.is_dir():
            print(f"⚠️  Path is not a directory: {directory_path}")
            return False
        
        # Remove all contents
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
                print(f"✓ Deleted file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"✓ Deleted directory: {item}")
        
        print(f"✓ Successfully cleared: {directory_path}\n")
        return True
    
    except PermissionError:
        print(f"✗ Permission denied: {directory_path}\n")
        return False
    except Exception as e:
        print(f"✗ Error clearing {directory_path}: {e}\n")
        return False


def clear_multiple_directories(directories):
    """
    Clear multiple directories.
    
    Args:
        directories (list): List of directory paths to clear
    """
    print("=" * 50)
    print("Directory Cleanup Started")
    print("=" * 50 + "\n")
    
    successful = 0
    failed = 0
    
    for directory in directories:
        if clear_directory(directory):
            successful += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"Cleanup Complete | Success: {successful} | Failed: {failed}")
    print("=" * 50)


if __name__ == "__main__":
    # Define directories to clear
    directories_to_clear = [
        "./logs",
        "./runs",
        "./videos"
    ]
    
    # Clear the directories
    clear_multiple_directories(directories_to_clear)
