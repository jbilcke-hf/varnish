import os
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def setup_debug_logging():
    """Configure detailed debug logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def verify_model_paths(model_paths: Dict[str, str], base_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Verify and convert relative model paths to absolute paths.
    
    Args:
        model_paths: Dictionary of model names to paths
        base_dir: Optional base directory to resolve relative paths against
        
    Returns:
        Dictionary with verified absolute paths
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"Base directory for models: {base_dir}")
    
    verified_paths = {}
    for name, path in model_paths.items():
        # Convert to Path object
        path_obj = Path(path)
        
        # If path is relative, make it absolute using base_dir
        if not path_obj.is_absolute():
            path_obj = base_dir / path_obj
        
        # Remove any './' prefix for consistency in logging
        try:
            path_obj = path_obj.resolve()
        except Exception as e:
            logger.error(f"Error resolving path for {name}: {e}")
            raise
            
        # Log existence check
        if path_obj.exists():
            logger.debug(f"✓ Model file exists: {name} -> {path_obj}")
        else:
            logger.error(f"✗ Model file missing: {name} -> {path_obj}")
            parent = path_obj.parent
            if parent.exists():
                logger.debug(f"Parent directory exists: {parent}")
                logger.debug(f"Contents of {parent}:")
                for item in parent.iterdir():
                    logger.debug(f"  {item.name}")
            else:
                logger.error(f"Parent directory missing: {parent}")
            
        verified_paths[name] = str(path_obj)
        
    return verified_paths

def log_directory_structure(start_path: str = ".", max_depth: int = 3):
    """
    Log the directory structure starting from a given path
    
    Args:
        start_path: Starting directory path
        max_depth: Maximum depth to traverse
    """
    start_path = Path(start_path).resolve()
    logger.debug(f"Directory structure from {start_path}:")
    
    def _log_directory(path: Path, depth: int = 0):
        if depth > max_depth:
            return
        try:
            for item in sorted(path.iterdir()):
                prefix = "  " * depth + ("📁 " if item.is_dir() else "📄 ")
                logger.debug(f"{prefix}{item.name}")
                if item.is_dir() and depth < max_depth:
                    _log_directory(item, depth + 1)
        except Exception as e:
            logger.error(f"Error accessing {path}: {e}")
    
    _log_directory(start_path)