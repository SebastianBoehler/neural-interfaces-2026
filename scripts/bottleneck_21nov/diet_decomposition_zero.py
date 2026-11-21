"""Apply the production component patch to the existing 379-window zero check."""
from diet_decomposition_patch import ARMS, patch
from returning_diet_zero import main


if __name__ == "__main__":
    main(arms=ARMS, source_patch=patch, setting="settings")
