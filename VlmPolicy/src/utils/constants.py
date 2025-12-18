"""
Constants and utilities for enums.
"""

from enum import Enum
from typing import Type, TypeVar, List


T = TypeVar('T', bound=Enum)


def add_enum_utilities(enum_cls: Type[T]) -> Type[T]:
    """
    Decorator to add utility methods to Enum classes.
    
    Adds:
    - values(): Returns list of all enum values
    - names(): Returns list of all enum names
    - from_string(s): Get enum from string (case-insensitive)
    
    Args:
        enum_cls: The Enum class to decorate
        
    Returns:
        The decorated Enum class with additional utility methods
    """
    
    @classmethod
    def values(cls) -> List[str]:
        """Get all enum values as a list."""
        return [member.value for member in cls]
    
    @classmethod
    def names(cls) -> List[str]:
        """Get all enum names as a list."""
        return [member.name for member in cls]
    
    @classmethod
    def from_string(cls, s: str, default=None):
        """
        Get enum member from string (case-insensitive).
        
        Args:
            s: String to match (can be name or value)
            default: Default value if no match found
            
        Returns:
            Enum member or default value
        """
        if not s:
            return default
        
        s_lower = s.lower()
        
        # Try matching by name
        for member in cls:
            if member.name.lower() == s_lower:
                return member
        
        # Try matching by value
        for member in cls:
            if str(member.value).lower() == s_lower:
                return member
        
        return default
    
    # Add methods to the enum class
    enum_cls.values = values
    enum_cls.names = names
    enum_cls.from_string = from_string
    
    return enum_cls

