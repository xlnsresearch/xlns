# src/xlnsconf/framework/common.py
import xlns as xl
import numpy as np

class LNSFrameworkBase:
    """Base class for framework-specific LNS implementations."""
    
    def __init__(self, data, precision=None):
        """
        Initialize with either numpy array, framework tensor, or existing xlns data.
        
        Args:
            data: Input data
            precision: Optional precision parameter (similar to xlnsv)
        """
        self._precision = precision or xl.xlnsF
        
        # Handle different input types
        if isinstance(data, xl.xlnsnp):
            self._lns_data = data
        else:
            # Convert to numpy first, then to xlns
            np_data = self._to_numpy(data)
            self._lns_data = xl.xlnsnp(np_data)
    
    def _to_numpy(self, data):
        """Convert data to numpy array."""
        if isinstance(data, np.ndarray):
            return data
        return np.array(data)
    
    @property
    def lns_data(self):
        """Access underlying LNS data."""
        return self._lns_data
    
    def __str__(self):
        return f"LNSFramework({self._lns_data})"
