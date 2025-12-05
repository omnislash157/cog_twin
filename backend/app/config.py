"""
Backend configuration - extends root config.yaml

Imports the main config system and adds UI-specific settings.
"""

import sys
from pathlib import Path

# Add project root to path so we can import the main config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg, get_config, get_data_dir, get_api_key


# =============================================================================
# UI-SPECIFIC SETTINGS (extend config.yaml in Phase 7.x)
# =============================================================================

class UISettings:
    """UI-specific settings with defaults."""
    
    @property
    def app_name(self) -> str:
        return cfg("ui.app_name", "CogTwin UI")
    
    @property
    def cors_origins(self) -> list[str]:
        return cfg("ui.cors_origins", ["http://localhost:5173", "http://localhost:3000"])
    
    @property
    def ws_heartbeat_interval(self) -> int:
        return cfg("ui.ws_heartbeat_interval", 30)
    
    @property
    def engine_path(self) -> Path:
        return PROJECT_ROOT
    
    @property
    def data_dir(self) -> Path:
        return get_data_dir()


# Singleton
settings = UISettings()


# Re-export for convenience
__all__ = [
    "settings",
    "cfg",
    "get_config", 
    "get_data_dir",
    "get_api_key",
    "PROJECT_ROOT",
]