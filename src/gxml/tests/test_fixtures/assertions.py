"""Custom assertions for GXML testing."""

import numpy as np


def assert_corner_points(test_case, panel, *expected, msg=None):
    """Assert panel corner points match expected world positions.
    
    Verifies that a panel's four corners, when transformed to world space, match
    expected coordinates. Uses numpy.allclose for floating point comparison with
    1e-6 tolerance. Provides detailed error messages indicating which corner
    failed and the actual vs expected values.
    
    Args:
        test_case: unittest.TestCase instance for assertions
        panel: Panel with transform_point method (typically GXMLPanel or subclass)
        *expected: Either:
            - 4 separate point arguments: p1, p2, p3, p4
            - 1 array/list of 4 points: [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]
        msg: Optional custom failure message (keyword-only)
    
    Expected points correspond to local corners:
        p1 = [0,0,0], p2 = [1,0,0], p3 = [1,1,0], p4 = [0,1,0]
    
    Raises:
        AssertionError: If any corner doesn't match expected position within tolerance
    
    Examples:
        Separate point arguments:
            >>> assert_corner_points(
            ...     self, panel,
            ...     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
            ... )
        
        Array of points:
            >>> assert_corner_points(
            ...     self, panel,
            ...     [[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]],
            ...     msg="Front face corners"
            ... )
    """
    # Handle both call signatures: 4 separate points or 1 array
    if len(expected) == 1:
        # Array of 4 points
        expected_points = np.array(expected[0])
        if expected_points.shape != (4, 3):
            raise ValueError(f"Expected array of 4 points with shape (4,3), got {expected_points.shape}")
        p1_expected, p2_expected, p3_expected, p4_expected = expected_points
    elif len(expected) == 4:
        # 4 separate points
        p1_expected, p2_expected, p3_expected, p4_expected = expected
    else:
        raise ValueError(f"Expected either 4 separate points or 1 array of 4 points, got {len(expected)} arguments")
    
    # Transform local corners to world space
    p1_actual = panel.transform_point([0, 0, 0])
    p2_actual = panel.transform_point([1, 0, 0])
    p3_actual = panel.transform_point([1, 1, 0])
    p4_actual = panel.transform_point([0, 1, 0])
    
    # Check each corner with detailed error messages
    corner_info = panel.subId if hasattr(panel, 'subId') else 'panel'
    test_case.assertTrue(np.allclose(p1_actual, np.array(p1_expected), atol=1e-6),
                        msg or f"{corner_info} corner [0,0,0]: expected {p1_expected}, got {p1_actual}")
    test_case.assertTrue(np.allclose(p2_actual, np.array(p2_expected), atol=1e-6),
                        msg or f"{corner_info} corner [1,0,0]: expected {p2_expected}, got {p2_actual}")
    test_case.assertTrue(np.allclose(p3_actual, np.array(p3_expected), atol=1e-6),
                        msg or f"{corner_info} corner [1,1,0]: expected {p3_expected}, got {p3_actual}")
    test_case.assertTrue(np.allclose(p4_actual, np.array(p4_expected), atol=1e-6),
                        msg or f"{corner_info} corner [0,1,0]: expected {p4_expected}, got {p4_actual}")


# Alias for backward compatibility
assert_face_corners = assert_corner_points

