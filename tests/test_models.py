import pytest
from tzd.models.lit_model import LitModel

def test_lit_model_creation():
    """
    Tests that the LitModel can be created with the correct dimensions.
    """
    model = LitModel(input_size=10, output_size=1)
    assert model.linear.in_features == 10
    assert model.linear.out_features == 1
