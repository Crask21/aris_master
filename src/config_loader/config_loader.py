"""
Simple YAML configuration loader with easy access and modification capabilities.
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigLoader:
    """
    A simple configuration loader for YAML files.
    
    Provides easy methods to:
    - Load entire or partial config sections
    - Get nested values using dot notation
    - Update config values
    - Save changes back to file
    
    Example:
        >>> config = ConfigLoader('config.yaml')
        >>> batch_size = config.get('data.batch_size')
        >>> config.set('data.batch_size', 32)
        >>> config.save()
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load the configuration from file.
        
        Returns:
            The loaded configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f) or {}
        
        return self._config
    
    def reload(self) -> Dict[str, Any]:
        """
        Reload the configuration from file (discards in-memory changes).
        
        Returns:
            The reloaded configuration dictionary
        """
        return self.load()
    
    def get(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation or return entire config.
        
        Args:
            key: Key in dot notation (e.g., 'data.batch_size'). 
                 If None, returns entire config.
            default: Default value to return if key is not found
        
        Returns:
            The configuration value or default
        
        Example:
            >>> config.get('data.batch_size')
            16
            >>> config.get('data.missing_key', default=10)
            10
            >>> config.get()  # Returns entire config
        """
        if key is None:
            return self._config
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire section of the config.
        
        Args:
            section: Top-level section name (e.g., 'data')
        
        Returns:
            The configuration section dictionary
        
        Example:
            >>> data_config = config.get_section('data')
        """
        return self.get(section, default={})
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Key in dot notation (e.g., 'data.batch_size')
            value: Value to set
        
        Example:
            >>> config.set('data.batch_size', 32)
            >>> config.set('new_section.new_key', 'value')
        """
        keys = key.split('.')
        current = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                raise ValueError(f"Cannot set nested key: '{k}' is not a dictionary")
            current = current[k]
        
        # Set the final key
        current[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values at once.
        
        Args:
            updates: Dictionary of keys (dot notation) and values
        
        Example:
            >>> config.update({
            ...     'data.batch_size': 32,
            ...     'data.learning_rate': 0.001
            ... })
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """
        Update an entire section of the config.
        
        Args:
            section: Section name (e.g., 'data')
            values: Dictionary of values to merge into the section
        
        Example:
            >>> config.update_section('data', {'batch_size': 32, 'epochs': 100})
        """
        if section not in self._config:
            self._config[section] = {}
        
        if not isinstance(self._config[section], dict):
            raise ValueError(f"Section '{section}' is not a dictionary")
        
        self._config[section].update(values)
    
    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the current configuration to file.
        
        Args:
            output_path: Optional path to save to. If None, saves to original file.
        
        Example:
            >>> config.save()  # Save to original file
            >>> config.save('new_config.yaml')  # Save to new file
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    def has(self, key: str) -> bool:
        """
        Check if a key exists in the configuration.
        
        Args:
            key: Key in dot notation (e.g., 'data.batch_size')
        
        Returns:
            True if the key exists, False otherwise
        
        Example:
            >>> config.has('data.batch_size')
            True
        """
        return self.get(key) is not None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the configuration.
        
        Args:
            key: Key in dot notation (e.g., 'data.batch_size')
        
        Returns:
            True if the key was deleted, False if it didn't exist
        
        Example:
            >>> config.delete('data.old_parameter')
            True
        """
        keys = key.split('.')
        current = self._config
        
        try:
            # Navigate to parent
            for k in keys[:-1]:
                current = current[k]
            
            # Delete the final key
            if keys[-1] in current:
                del current[keys[-1]]
                return True
            return False
        except (KeyError, TypeError):
            return False
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the configuration as a dictionary.
        
        Returns:
            Copy of the configuration dictionary
        """
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value using bracket notation.
        
        Args:
            key: Configuration key
        
        Returns:
            The configuration value
        
        Example:
            >>> config['data']['batch_size']
            16
            >>> config['data']
            {'batch_size': 16, 'epochs': 200, ...}
        """
        if key not in self._config:
            raise KeyError(f"Key '{key}' not found in configuration")
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set configuration value using bracket notation.
        
        Args:
            key: Configuration key
            value: Value to set
        
        Example:
            >>> config['data']['batch_size'] = 32
        """
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists using 'in' operator.
        
        Args:
            key: Configuration key
        
        Returns:
            True if key exists, False otherwise
        
        Example:
            >>> 'data' in config
            True
        """
        return key in self._config
    
    def __repr__(self) -> str:
        return f"ConfigLoader('{self.config_path}')"
    
    def __str__(self) -> str:
        return yaml.safe_dump(self._config, default_flow_style=False)


def load_config(config_path: Union[str, Path]) -> ConfigLoader:
    """
    Convenience function to create a ConfigLoader instance.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        ConfigLoader instance
    
    Example:
        >>> config = load_config('config.yaml')
    """
    return ConfigLoader(config_path)
