import numpy as np
import xlns as xl

def convert_to_lns(data, precision=None):
    """
    Convert data to LNS format, preserving precision.
    
    Args:
        data: Input data (numpy array or framework tensor)
        precision: Optional precision parameter
    
    Returns:
        xlnsnp: Data in LNS format
    """
    if isinstance(data, xl.xlnsnp):
        return data
        
    # Convert to numpy if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        
    # Use either global or specified precision
    old_f = xl.xlnsF
    if precision is not None:
        xl.xlnssetF(precision)
        
    try:
        result = xl.xlnsnp(data)
    finally:
        # Restore original precision
        if precision is not None:
            xl.xlnssetF(old_f)
            
    return result