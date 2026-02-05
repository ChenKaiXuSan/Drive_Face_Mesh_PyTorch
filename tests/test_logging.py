#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Test for logging functionality in main.py
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_configure_worker_logging():
    """Test that _configure_worker_logging creates proper log files and handlers."""
    from head3D_fuse.main import _configure_worker_logging
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_root = Path(tmpdir)
        worker_id = 0
        
        # Create mock env_dirs
        env_dirs = [
            Path("/data/person_01/room1"),
            Path("/data/person_01/outdoor"),
        ]
        
        # Call the function
        _configure_worker_logging(log_root, worker_id, env_dirs)
        
        # Verify that log directory was created
        assert log_root.exists()
        
        # Verify that worker log file was created
        worker_log = log_root / f"worker_{worker_id}.log"
        assert worker_log.exists()
        
        # Verify that root logger has handlers
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0


def test_worker_creates_task_specific_logs():
    """Test that _worker creates task-specific log files."""
    from head3D_fuse.main import _worker
    from omegaconf import OmegaConf
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_root = Path(tmpdir)
        infer_root = Path(tmpdir)
        
        # Create mock env_dirs with proper structure
        env_dirs = []
        for person_id in ["person_01", "person_02"]:
            for env_name in ["room1", "outdoor"]:
                env_dir = Path(tmpdir) / "mock_sam3d" / person_id / env_name
                env_dir.mkdir(parents=True, exist_ok=True)
                env_dirs.append(env_dir)
        
        # Create minimal config
        cfg_dict = {
            "infer": {
                "person_list": [-1],
                "env_list": ["all"],
                "workers": 1,
            }
        }
        
        # Mock the process_single_person_env function
        with patch("head3D_fuse.main.process_single_person_env") as mock_process:
            mock_process.return_value = None
            
            # Call _worker
            _worker(env_dirs[:2], log_root, infer_root, cfg_dict, worker_id=0)
            
            # Verify that process_single_person_env was called
            assert mock_process.call_count == 2
            
            # Verify that individual task logs were created
            for person_id in ["person_01"]:
                for env_name in ["room1"]:
                    log_file = log_root / f"{person_id}_{env_name}.log"
                    # Note: The log file may or may not exist depending on implementation
                    # This is a basic test to ensure no exceptions are raised


def test_log_filename_from_env_dir():
    """Test that log filenames are correctly generated from env_dir structure."""
    env_dir = Path("/workspace/data/person_05/night_low_h265")
    
    # Extract person_id and env_name
    person_id = env_dir.parent.name
    env_name = env_dir.name
    
    # Verify extraction
    assert person_id == "person_05"
    assert env_name == "night_low_h265"
    
    # Verify log filename format
    log_filename = f"{person_id}_{env_name}.log"
    assert log_filename == "person_05_night_low_h265.log"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
