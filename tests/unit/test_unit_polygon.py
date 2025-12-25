"""
Unit tests for GXMLPolygon
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from elements.gxml_polygon import GXMLPolygon


class PolygonUnitTests(unittest.TestCase):
    """Unit tests for GXMLPolygon element."""
    
    def test_create_triangle(self):
        """Test creating a triangle."""
        poly = GXMLPolygon.triangle([0, 0, 0], [1, 0, 0], [0.5, 1, 0])
        
        self.assertEqual(poly.vertex_count, 3)
        np.testing.assert_array_equal(poly.get_vertex(0), [0, 0, 0])
        np.testing.assert_array_equal(poly.get_vertex(1), [1, 0, 0])
        np.testing.assert_array_equal(poly.get_vertex(2), [0.5, 1, 0])
    
    def test_create_quad(self):
        """Test creating a quad."""
        poly = GXMLPolygon.quad([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])
        
        self.assertEqual(poly.vertex_count, 4)
    
    def test_vertex_wrapping(self):
        """Test that vertex indices wrap around."""
        poly = GXMLPolygon.triangle([0, 0, 0], [1, 0, 0], [0.5, 1, 0])
        
        # Index 3 should wrap to 0
        np.testing.assert_array_equal(poly.get_vertex(3), poly.get_vertex(0))
        np.testing.assert_array_equal(poly.get_vertex(-1), poly.get_vertex(2))
    
    def test_normal_computation(self):
        """Test that normal is computed correctly for CCW vertices."""
        # Triangle in XY plane, CCW when viewed from +Z
        poly = GXMLPolygon.triangle([0, 0, 0], [1, 0, 0], [0, 1, 0])
        
        normal = poly.normal
        # Normal should point in +Z direction
        self.assertAlmostEqual(normal[2], 1.0, places=5)
        self.assertAlmostEqual(normal[0], 0.0, places=5)
        self.assertAlmostEqual(normal[1], 0.0, places=5)
    
    def test_explicit_normal(self):
        """Test setting an explicit normal."""
        poly = GXMLPolygon.triangle([0, 0, 0], [1, 0, 0], [0, 1, 0])
        poly.normal = [0, 0, -1]  # Override computed normal
        
        np.testing.assert_array_almost_equal(poly.normal, [0, 0, -1])
    
    def test_edge_access(self):
        """Test getting edge start/end points."""
        poly = GXMLPolygon.quad([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0])
        
        start, end = poly.get_edge(0)
        np.testing.assert_array_equal(start, [0, 0, 0])
        np.testing.assert_array_equal(end, [1, 0, 0])
        
        # Edge 3 connects vertex 3 back to vertex 0
        start, end = poly.get_edge(3)
        np.testing.assert_array_equal(start, [0, 1, 0])
        np.testing.assert_array_equal(end, [0, 0, 0])
    
    def test_point_on_edge(self):
        """Test parametric point along an edge."""
        poly = GXMLPolygon.quad([0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0])
        
        # Midpoint of edge 0 (from [0,0,0] to [2,0,0])
        mid = poly.point_on_edge(0, 0.5)
        np.testing.assert_array_almost_equal(mid, [1, 0, 0])
        
        # Quarter point
        quarter = poly.point_on_edge(0, 0.25)
        np.testing.assert_array_almost_equal(quarter, [0.5, 0, 0])
    
    def test_centroid(self):
        """Test centroid calculation."""
        poly = GXMLPolygon.quad([0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0])
        
        centroid = poly.get_centroid()
        np.testing.assert_array_almost_equal(centroid, [1, 1, 0])
    
    def test_triangle_area(self):
        """Test area calculation for a triangle."""
        # Right triangle with legs of length 2
        poly = GXMLPolygon.triangle([0, 0, 0], [2, 0, 0], [0, 2, 0])
        
        area = poly.get_area()
        self.assertAlmostEqual(area, 2.0, places=5)  # 0.5 * 2 * 2 = 2
    
    def test_quad_area(self):
        """Test area calculation for a quad."""
        # 2x2 square
        poly = GXMLPolygon.quad([0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0])
        
        area = poly.get_area()
        self.assertAlmostEqual(area, 4.0, places=5)
    
    def test_perimeter(self):
        """Test perimeter calculation."""
        # 2x2 square
        poly = GXMLPolygon.quad([0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0])
        
        perimeter = poly.get_perimeter()
        self.assertAlmostEqual(perimeter, 8.0, places=5)
    
    def test_add_vertex(self):
        """Test adding vertices one at a time."""
        poly = GXMLPolygon()
        poly.add_vertex([0, 0, 0])
        poly.add_vertex([1, 0, 0])
        poly.add_vertex([0.5, 1, 0])
        
        self.assertEqual(poly.vertex_count, 3)
    
    def test_set_vertices(self):
        """Test setting all vertices at once."""
        poly = GXMLPolygon()
        poly.set_vertices([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        
        self.assertEqual(poly.vertex_count, 4)
    
    def test_pentagon(self):
        """Test creating an arbitrary 5-sided polygon."""
        # Regular pentagon-ish shape
        vertices = [
            [0, 0, 0],
            [2, 0, 0],
            [2.5, 1.5, 0],
            [1, 2.5, 0],
            [-0.5, 1.5, 0]
        ]
        poly = GXMLPolygon(vertices)
        
        self.assertEqual(poly.vertex_count, 5)
        self.assertIsNotNone(poly.normal)
    
    def test_barycentric_weights(self):
        """Test point from barycentric-style weights."""
        poly = GXMLPolygon.triangle([0, 0, 0], [3, 0, 0], [0, 3, 0])
        
        # Equal weights = centroid
        centroid = poly.point_from_barycentric([1, 1, 1])
        np.testing.assert_array_almost_equal(centroid, [1, 1, 0])
        
        # All weight on first vertex
        v0 = poly.point_from_barycentric([1, 0, 0])
        np.testing.assert_array_almost_equal(v0, [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
