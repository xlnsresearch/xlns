import pytest
import torch
import xlnstorch as xltorch

@pytest.fixture(scope="session")
def default_base():
    """Create a default base tensor for testing."""
    return torch.tensor(2.0 ** (2.0 ** -23), dtype=torch.float64)

@pytest.fixture
def sample_data():
    """Create sample data tensors for testing."""
    return {
        'positive': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        'negative': torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64),
        'mixed': torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=torch.float64),
        'zero': torch.tensor([0.0], dtype=torch.float64),
        'small': torch.tensor([1e-10, 1e-5], dtype=torch.float64),
        'large': torch.tensor([1e5, 1e10], dtype=torch.float64),
    }

@pytest.fixture
def sample_lns_tensors(sample_data, default_base):
    """Create sample LNS tensors for testing."""
    return {
        name: xltorch.lnstensor(data, b=default_base)
        for name, data in sample_data.items()
    }

def pytest_configure(config):
    """Configure pytest."""
    # Set default tensor type to double precision
    torch.set_default_dtype(torch.float64)
    
    # Add custom markers
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "gradient: marks tests that check gradient computation"
    ) 