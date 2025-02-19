"""Framework integration for xlns."""
from enum import Enum

class FrameworkType(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"

# Will be set based on which framework is being used
CURRENT_FRAMEWORK = None