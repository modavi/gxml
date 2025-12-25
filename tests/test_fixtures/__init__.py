"""Test fixtures and utilities for GXML testing.

Organized into logical modules:
- mocks: Mock classes for testing (GXMLMockPanel, GXMLTestRenderContext, GXMLMockLayout, LayoutPass)
- base_integration_test: Base class for integration tests (BaseIntegrationTest)
- assertions: Custom assertion functions (assert_corner_points, assert_face_corners)
"""

from .mocks import GXMLMockPanel, GXMLTestRenderContext, LayoutPass, GXMLMockLayout
from .base_integration_test import BaseIntegrationTest
from .assertions import assert_corner_points, assert_face_corners

__all__ = [
    'GXMLMockPanel',
    'GXMLTestRenderContext',
    'LayoutPass',
    'GXMLMockLayout',
    'BaseIntegrationTest',
    'assert_corner_points',
    'assert_face_corners',
]
