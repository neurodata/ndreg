# testlib.testcase.py

from unittest import TestCase


class BaseTestCase(TestCase):
    """
    All test cases should inherit from this class as any common
    functionality that is added here will then be available to all
    subclasses. This facilitates the ability to update in one spot
    and allow all tests to get the update for easy maintenance.
    """
    pass
