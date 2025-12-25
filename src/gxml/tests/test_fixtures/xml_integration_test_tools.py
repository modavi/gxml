"""
XML-based integration test tools for generating and validating test expectations.

Provides utilities to convert parsed panel structures into XML format for test
expectations and validate actual output against expected XML structures.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np


class XMLIntegrationTestGenerator:
    """Generate XML test expectations from parsed panel structures.
    
    Converts panels and their dynamic children (regions/faces) into XML format
    that captures corner points and hierarchy for test validation.
    
    Example:
        >>> generator = XMLIntegrationTestGenerator()
        >>> xml_output = generator.generate_from_panels(root.children)
        >>> print(xml_output)
        <r id="0">
          <r id="front" pts="0,0,0.5|1,0,0.5|1,1,0.5|0,1,0.5"/>
        </r>
    """
    
    def __init__(self, precision=6):
        """Initialize the generator.
        
        Args:
            precision: Number of decimal places for coordinate formatting (default: 6)
        """
        self.precision = precision
    
    def generate_from_panels(self, panels):
        """Generate XML structure from a list of panels.
        
        Args:
            panels: List of panel elements to convert to XML
            
        Returns:
            Pretty-printed XML string representing the panel structure
            
        Example:
            >>> xml = generator.generate_from_panels(root.children)
        """
        root_elem = ET.Element("root")
        
        for panel in panels:
            panel_elem = self._create_panel_element(panel)
            root_elem.append(panel_elem)
        
        return self._prettify_xml(root_elem)
    
    def generate_from_panel_xml(self, xml_string):
        """Parse panel XML, process it, and generate test expectation XML.
        
        This is a convenience method that handles the full pipeline:
        parsing, layout, and XML generation.
        
        Args:
            xml_string: XML string defining panels to process
            
        Returns:
            Pretty-printed XML string with panel corner points
            
        Example:
            >>> xml_input = '<root><panel thickness="0.1"/></root>'
            >>> xml_output = generator.generate_from_panel_xml(xml_input)
        """
        from gxml_parser import GXMLParser
        from gxml_layout import GXMLLayout
        from gxml_render import GXMLRender
        from tests.test_fixtures.mocks import GXMLTestRenderContext
        
        # Parse and process the XML
        root = GXMLParser.parse(xml_string)
        GXMLLayout.layout(root)
        render_context = GXMLTestRenderContext()
        GXMLRender.render(root, render_context)
        
        # Generate test XML from the processed panels
        return self.generate_from_panels(root.children)
    
    def _create_panel_element(self, panel):
        """Create XML element for a panel and its children.
        
        Args:
            panel: Panel element to convert
            
        Returns:
            ET.Element representing the panel
        """
        panel_elem = ET.Element("r")
        panel_elem.set("id", str(panel.id))
        
        # Get corner points for the panel itself
        corner_points = self._get_corner_points(panel)
        if corner_points:
            pts_str = self._format_points(corner_points)
            panel_elem.set("pts", pts_str)
        
        # Process nested panel children (hierarchical structure)
        if hasattr(panel, 'children') and panel.children:
            for child in panel.children:
                child_elem = self._create_panel_element(child)
                panel_elem.append(child_elem)
        
        # Process dynamic children (regions/faces)
        if hasattr(panel, 'dynamicChildren') and panel.dynamicChildren:
            for child in panel.dynamicChildren:
                child_elem = self._create_region_element(child)
                panel_elem.append(child_elem)
        
        return panel_elem
    
    def _create_region_element(self, region):
        """Create XML element for a region/face with corner points.
        
        Args:
            region: Region or face element to convert
            
        Returns:
            ET.Element representing the region with corner points
        """
        # Check if this is a polygon (GXMLPolygon has vertices property)
        is_polygon = hasattr(region, 'vertices') and hasattr(region, 'vertex_count')
        
        region_elem = ET.Element("r")
        
        # Add type attribute for polygons
        if is_polygon:
            region_elem.set("type", "polygon")
        
        # Set ID attribute (use subId if available, otherwise use id)
        region_id = getattr(region, 'subId', None) or getattr(region, 'id', 'unknown')
        region_elem.set("id", str(region_id))
        
        # Get corner points - use polygon vertices directly if available
        if is_polygon:
            corner_points = self._get_polygon_points(region)
        else:
            corner_points = self._get_corner_points(region)
        
        if corner_points:
            pts_str = self._format_points(corner_points)
            region_elem.set("pts", pts_str)
        
        # Recursively process any dynamic children
        if hasattr(region, 'dynamicChildren') and region.dynamicChildren:
            for child in region.dynamicChildren:
                child_elem = self._create_region_element(child)
                region_elem.append(child_elem)
        
        return region_elem
    
    def _get_polygon_points(self, polygon):
        """Extract vertices from a polygon element.
        
        Args:
            polygon: Polygon element with vertices property
            
        Returns:
            List of vertex points (in world space if transform available)
        """
        # Use get_world_vertices if available (transforms to world space)
        if hasattr(polygon, 'get_world_vertices'):
            return polygon.get_world_vertices()
        
        # Otherwise return raw vertices
        return polygon.vertices
    
    def _get_corner_points(self, element):
        """Extract corner points from an element.
        
        Args:
            element: Element to extract corner points from
            
        Returns:
            List of corner points as numpy arrays, or None if not available
        """
        # Try to get corner points using element's transform_point method
        # This handles quad interpolation for variable heights
        if not hasattr(element, 'transform_point'):
            # Fall back to transform.transform_point for elements without quad
            if not hasattr(element, 'transform'):
                return None
            transform_func = element.transform.transform_point
        else:
            transform_func = element.transform_point
        
        try:
            # Standard quad corners in local space
            local_corners = [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ]
            
            # Transform to world space
            world_corners = [
                transform_func(corner)
                for corner in local_corners
            ]
            
            return world_corners
        except Exception:
            return None
    
    def _format_points(self, points):
        """Format list of points into pipe-separated string.
        
        Args:
            points: List of 3D points (tuples or numpy arrays)
            
        Returns:
            String formatted as "x1,y1,z1|x2,y2,z2|..."
        """
        formatted_points = []
        for point in points:
            # Convert to numpy array if needed
            if not isinstance(point, np.ndarray):
                point = np.array(point)
            
            # Format coordinates with specified precision
            coords = [f"{coord:.{self.precision}g}" for coord in point]
            formatted_points.append(",".join(coords))
        
        return "|".join(formatted_points)
    
    def _prettify_xml(self, elem):
        """Return pretty-printed XML string.
        
        Args:
            elem: ET.Element to prettify
            
        Returns:
            Indented XML string without XML declaration
        """
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty = reparsed.toprettyxml(indent="    ")
        
        # Remove XML declaration line
        lines = pretty.split('\n')
        if lines and lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        return '\n'.join(lines)


class XMLIntegrationTestValidator:
    """Validate actual panel output against expected XML structures.
    
    Compares generated panel structures with expected test XML, checking
    corner points, IDs, and hierarchy.
    
    Example:
        >>> validator = XMLIntegrationTestValidator()
        >>> validator.validate(expected_xml, actual_panels)
    """
    
    def __init__(self, tolerance=1e-6):
        """Initialize the validator.
        
        Args:
            tolerance: Numerical tolerance for point comparisons (default: 1e-6)
        """
        self.tolerance = tolerance
        self.errors = []
    
    def validate(self, expected_xml, actual_panels, expected_root_name="root"):
        """Validate actual panels against expected XML.
        
        Args:
            expected_xml: XML string defining expected structure and points
            actual_panels: List of actual panel elements to validate
            expected_root_name: Expected name of root element (default: "root")
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            AssertionError: If validation fails with detailed error message
        """
        self.errors = []
        
        # Parse expected XML
        root = ET.fromstring(expected_xml)
        
        # Validate root element name
        if root.tag != expected_root_name:
            self.errors.append(
                f"Root element name mismatch: expected '{expected_root_name}', "
                f"got '{root.tag}'"
            )
            return self._raise_validation_error()
        
        expected_panels = list(root)
        
        # Check panel count
        if len(expected_panels) != len(actual_panels):
            self.errors.append(
                f"Panel count mismatch: expected {len(expected_panels)}, "
                f"got {len(actual_panels)}"
            )
            return self._raise_validation_error()
        
        # Validate each panel
        for expected_panel, actual_panel in zip(expected_panels, actual_panels):
            self._validate_panel(expected_panel, actual_panel)
        
        if self.errors:
            return self._raise_validation_error()
        
        return True
    
    def _validate_panel(self, expected_elem, actual_panel):
        """Validate a single panel.
        
        Args:
            expected_elem: ET.Element with expected panel data
            actual_panel: Actual panel element to validate
        """
        # Validate ID
        expected_id = expected_elem.get("id")
        actual_id = str(actual_panel.id)
        if expected_id != actual_id:
            self.errors.append(
                f"Panel ID mismatch: expected '{expected_id}', got '{actual_id}'"
            )
        
        # Validate panel corner points if present
        expected_pts = expected_elem.get("pts")
        if expected_pts:
            self._validate_points(expected_pts, actual_panel, actual_id, "panel")
        
        # Validate child elements (both nested panels and dynamic children)
        expected_children = list(expected_elem)
        actual_children = []
        
        # Add nested panels first
        if hasattr(actual_panel, 'children') and actual_panel.children:
            actual_children.extend(actual_panel.children)
        
        # Then add dynamic children (regions/faces)
        if hasattr(actual_panel, 'dynamicChildren') and actual_panel.dynamicChildren:
            actual_children.extend(actual_panel.dynamicChildren)
        
        if len(expected_children) != len(actual_children):
            self.errors.append(
                f"Panel '{actual_id}' child count mismatch: "
                f"expected {len(expected_children)}, got {len(actual_children)}"
            )
            return
        
        for expected_child, actual_child in zip(expected_children, actual_children):
            # Check if this is a nested panel or a region/face
            # A region has subId, a nested panel doesn't (or has empty subId)
            is_region = hasattr(actual_child, 'subId') and actual_child.subId
            if is_region:
                # It's a region/face - check type attribute to determine if polygon
                expected_type = expected_child.get("type")
                is_polygon = expected_type == "polygon"
                self._validate_region(expected_child, actual_child, actual_id, is_polygon=is_polygon)
            else:
                # It's a nested panel, validate as panel
                self._validate_panel(expected_child, actual_child)
    
    def _validate_region(self, expected_elem, actual_region, panel_id, is_polygon=False):
        """Validate a single region/face.
        
        Args:
            expected_elem: ET.Element with expected region data
            actual_region: Actual region element to validate
            panel_id: Parent panel ID for error messages
            is_polygon: Whether this is a polygon element (uses vertices directly)
        """
        # Validate region ID
        expected_id = expected_elem.get("id")
        actual_id = str(getattr(actual_region, 'subId', None) or 
                       getattr(actual_region, 'id', 'unknown'))
        
        if expected_id != actual_id:
            self.errors.append(
                f"Panel '{panel_id}' region ID mismatch: "
                f"expected '{expected_id}', got '{actual_id}'"
            )
        
        # Validate corner points if present
        expected_pts = expected_elem.get("pts")
        if expected_pts:
            self._validate_points(expected_pts, actual_region, panel_id, actual_id, is_polygon=is_polygon)
        
        # Recursively validate children
        expected_children = list(expected_elem)
        actual_children = getattr(actual_region, 'dynamicChildren', [])
        
        if len(expected_children) != len(actual_children):
            self.errors.append(
                f"Panel '{panel_id}' region '{actual_id}' child count mismatch: "
                f"expected {len(expected_children)}, got {len(actual_children)}"
            )
            return
        
        for expected_child, actual_child in zip(expected_children, actual_children):
            child_expected_type = expected_child.get("type")
            child_is_polygon = child_expected_type == "polygon"
            self._validate_region(expected_child, actual_child, panel_id, is_polygon=child_is_polygon)
    
    def _validate_points(self, expected_pts_str, actual_region, panel_id, region_id, is_polygon=False):
        """Validate corner points match expected values.
        
        Args:
            expected_pts_str: Pipe-separated point string
            actual_region: Actual region to extract points from
            panel_id: Parent panel ID for error messages
            region_id: Region ID for error messages
            is_polygon: Whether this is a polygon element (uses vertices directly)
        """
        # Parse expected points
        expected_points = self._parse_points(expected_pts_str)
        
        # Get actual points - use polygon vertices or quad corners
        if is_polygon:
            actual_points = self._get_polygon_points(actual_region)
        else:
            actual_points = self._get_actual_points(actual_region)
        
        if actual_points is None:
            self.errors.append(
                f"Panel '{panel_id}' region '{region_id}': "
                f"Could not extract corner points from actual region"
            )
            return
        
        if len(expected_points) != len(actual_points):
            self.errors.append(
                f"Panel '{panel_id}' region '{region_id}' point count mismatch: "
                f"expected {len(expected_points)}, got {len(actual_points)}"
            )
            return
        
        # Compare each point
        for i, (expected_pt, actual_pt) in enumerate(zip(expected_points, actual_points)):
            if not np.allclose(expected_pt, actual_pt, atol=self.tolerance):
                self.errors.append(
                    f"Panel '{panel_id}' region '{region_id}' point {i} mismatch:\n"
                    f"  expected: {expected_pt}\n"
                    f"  got:      {actual_pt}\n"
                    f"  diff:     {np.abs(expected_pt - actual_pt)}"
                )
    
    def _parse_points(self, pts_str):
        """Parse pipe-separated point string into list of numpy arrays.
        
        Args:
            pts_str: String formatted as "x1,y1,z1|x2,y2,z2|..."
            
        Returns:
            List of numpy arrays representing 3D points
        """
        points = []
        for pt_str in pts_str.split("|"):
            coords = [float(x) for x in pt_str.split(",")]
            points.append(np.array(coords))
        return points
    
    def _get_actual_points(self, region):
        """Extract corner points from actual region.
        
        Args:
            region: Region element to extract points from
            
        Returns:
            List of numpy arrays, or None if extraction fails
        """
        # Use element's transform_point method if available (handles quad interpolation)
        if not hasattr(region, 'transform_point'):
            # Fall back to transform.transform_point for elements without quad
            if not hasattr(region, 'transform'):
                return None
            transform_func = region.transform.transform_point
        else:
            transform_func = region.transform_point
        
        try:
            local_corners = [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0)
            ]
            
            world_corners = [
                transform_func(corner)
                for corner in local_corners
            ]
            
            return world_corners
        except Exception:
            return None
    
    def _get_polygon_points(self, polygon):
        """Extract vertices from a polygon element.
        
        Args:
            polygon: Polygon element with vertices property
            
        Returns:
            List of numpy arrays, or None if extraction fails
        """
        # Use get_world_vertices if available (transforms to world space)
        if hasattr(polygon, 'get_world_vertices'):
            return polygon.get_world_vertices()
        
        # Otherwise return raw vertices
        if hasattr(polygon, 'vertices'):
            return polygon.vertices
        
        return None
    
    def _raise_validation_error(self):
        """Raise assertion error with all accumulated errors."""
        error_msg = "Validation failed:\n" + "\n".join(f"  - {err}" for err in self.errors)
        raise AssertionError(error_msg)
    
    def get_errors(self):
        """Get list of validation errors.
        
        Returns:
            List of error message strings
        """
        return self.errors.copy()
