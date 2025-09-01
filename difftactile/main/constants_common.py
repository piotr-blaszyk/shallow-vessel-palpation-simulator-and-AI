import json
from typing import Any, Dict, Optional


class ConstantsFromJson:
    def __init__(self, json_path: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Initialize constants from either a JSON file path or a dictionary.
        
        Args:
            json_path: Path to JSON file (optional)
            data: Dictionary containing data (optional)
        """
        if json_path is not None:
            with open(json_path, 'r') as f:
                data = json.load(f)
        
        if data is None:
            return
            
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConstantsFromJson(data=value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
    
    def __repr__(self) -> str:
        """Return a string representation of the object's attributes."""
        attrs = [f"{key}={repr(value)}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"


class ConstantsManual:
    def __init__(self):
        pass