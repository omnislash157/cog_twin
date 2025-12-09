"""
Config Loader - Single source of truth for feature flags.

Loads enterprise_config.yaml and provides helpers for checking feature states.

Usage:
    from config_loader import load_config, cfg, memory_enabled
    
    load_config("enterprise_config.yaml")
    
    if memory_enabled():
        # Full pipeline
    else:
        # Context stuffing only

Version: 1.0.0
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Global config cache
_config: Dict[str, Any] = {}
_loaded: bool = False


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from yaml file.
    
    Args:
        path: Path to config file. If None, tries:
              1. COGTWIN_CONFIG env var
              2. enterprise_config.yaml
              3. config.yaml
              
    Returns:
        Loaded config dict
    """
    global _config, _loaded
    
    if path is None:
        # Try env var first
        path = os.environ.get("COGTWIN_CONFIG")
        
        if not path:
            # Try enterprise config
            if Path("enterprise_config.yaml").exists():
                path = "enterprise_config.yaml"
            elif Path("config.yaml").exists():
                path = "config.yaml"
            else:
                raise FileNotFoundError(
                    "No config found. Set COGTWIN_CONFIG or create enterprise_config.yaml"
                )
    
    with open(path) as f:
        _config = yaml.safe_load(f)
    
    _loaded = True
    return _config


def cfg(key: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Examples:
        cfg('features.memory_pipelines')  # -> False
        cfg('deployment.tier')             # -> 'basic'
        cfg('model.temperature', 0.7)      # -> 0.5 or default
        
    Args:
        key: Dot-separated path to config value
        default: Default if not found
        
    Returns:
        Config value or default
    """
    if not _loaded:
        load_config()
    
    keys = key.split('.')
    val = _config
    
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    
    return val


def get_config() -> Dict[str, Any]:
    """Get the full config dict."""
    if not _loaded:
        load_config()
    return _config


# =============================================================================
# CONVENIENCE HELPERS
# =============================================================================

def is_enterprise_mode() -> bool:
    """Check if running in enterprise (multi-tenant) mode."""
    return cfg('deployment.mode') == 'enterprise'


def is_personal_mode() -> bool:
    """Check if running in personal (single user) mode."""
    return cfg('deployment.mode', 'personal') == 'personal'


def memory_enabled() -> bool:
    """Check if memory pipelines are enabled."""
    return cfg('features.memory_pipelines', True)


def context_stuffing_enabled() -> bool:
    """Check if context stuffing is enabled."""
    return cfg('features.context_stuffing', False)


def get_tier() -> str:
    """Get deployment tier (basic, advanced, full)."""
    return cfg('deployment.tier', 'full')


def get_allowed_domains() -> list:
    """Get list of allowed email domains."""
    return cfg('deployment.auth.allowed_domains', [])


def get_ui_features() -> Dict[str, bool]:
    """
    Get UI feature flags to pass to frontend.
    
    Returns dict of feature_name -> enabled
    """
    defaults = {
        'swarm_loop': True,
        'memory_space_3d': True,
        'chat_basic': True,
        'dark_mode': True,
    }
    return cfg('features.ui', defaults)


def get_division_voice(division: str) -> str:
    """Get voice template name for a division."""
    mapping = cfg('voice.division_voice', {})
    return mapping.get(division, cfg('voice.default', 'corporate'))


# =============================================================================
# TIER PRESETS
# =============================================================================

TIER_PRESETS = {
    'basic': {
        'features.memory_pipelines': False,
        'features.context_stuffing': True,
        'features.ui.swarm_loop': False,
        'features.ui.memory_space_3d': False,
    },
    'advanced': {
        'features.memory_pipelines': True,
        'features.context_stuffing': False,
        'features.ui.swarm_loop': False,
        'features.ui.memory_space_3d': False,
    },
    'full': {
        'features.memory_pipelines': True,
        'features.context_stuffing': True,
        'features.ui.swarm_loop': True,
        'features.ui.memory_space_3d': True,
    },
}


def apply_tier_preset(tier: str):
    """
    Apply tier preset overrides to config.
    
    Useful for testing different configurations.
    """
    global _config
    
    if tier not in TIER_PRESETS:
        return
    
    preset = TIER_PRESETS[tier]
    
    for key, value in preset.items():
        keys = key.split('.')
        target = _config
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        config = load_config(config_path)
        print(f"Config loaded: {config_path or 'auto-detected'}")
        print()
        
        print("Deployment:")
        print(f"  Mode: {cfg('deployment.mode')}")
        print(f"  Tier: {cfg('deployment.tier')}")
        print(f"  Allowed domains: {get_allowed_domains()}")
        print()
        
        print("Features:")
        print(f"  Memory pipelines: {memory_enabled()}")
        print(f"  Context stuffing: {context_stuffing_enabled()}")
        print()
        
        print("UI Features:")
        for feature, enabled in get_ui_features().items():
            print(f"  {feature}: {enabled}")
        print()
        
        print("Model:")
        print(f"  Provider: {cfg('model.provider')}")
        print(f"  Name: {cfg('model.name')}")
        print(f"  Context window: {cfg('model.context_window'):,}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

# =============================================================================
# DOCS CONFIG HELPERS
# =============================================================================

def get_division_categories(division: str) -> list:
    """
    Get document categories accessible to a division.

    Args:
        division: Division name (e.g., "warehouse", "hr")

    Returns:
        List of doc category folder names this division can access
    """
    mapping = cfg('docs.division_categories', {})
    return mapping.get(division, mapping.get('default', ['general']))


def get_docs_dir() -> str:
    """Get the documents directory path."""
    return cfg('docs.docs_dir', './manuals')


def get_max_stuffing_tokens() -> int:
    """Get max tokens to stuff per request."""
    return cfg('docs.stuffing.max_tokens_per_division', 200000)

