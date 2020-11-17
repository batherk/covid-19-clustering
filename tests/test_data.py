# type: ignore
import pathsetup  # noqa
from src.utils import mapIndexToLocation


def test_data():
    assert mapIndexToLocation(145) == 'Norway'
