"""Share current CSP covariance until the native recognition latch retires the alternate."""

from shared_cov import SharedCovSelector


class SharedCovRetiringSelector(SharedCovSelector):
    retire_on_recognition = True
