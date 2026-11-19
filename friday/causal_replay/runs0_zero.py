"""Prove the independent exact runs0 terminal archive before the matched fits."""

from runs0_case import ORDER, PINS
from within0_complete import archive_zero


if __name__ == "__main__":
    archive_zero(ORDER, PINS, rest_seeds=3)
