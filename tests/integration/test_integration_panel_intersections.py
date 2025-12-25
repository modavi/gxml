"""
Integration tests for panel intersections (T-junctions, L-junctions, etc.)
"""
import unittest
from tests.test_fixtures.base_integration_test import BaseIntegrationTest


class JointTests(BaseIntegrationTest):
    """Tests for joint (L-junction) panel intersections with mitered corners."""

    def testBasicLJunction(self):
        """Test a basic L-junction where panel 2 rotates 90° at the end of panel 1.
        
        Panel 0 extends along X from 0 to 1. Panel 1 rotates 90° and attaches
        at the end of panel 0, extending into -Z direction.
        
        This creates an L-shaped corner joint with mitered faces:
        - Panel 0's END face is suppressed (interior geometry)
        - Panel 1's START face is suppressed (interior geometry)
        - FRONT faces overshoot to meet at corner (1.05, y, 0.05)
        - BACK faces meet at (0.95, y, -0.05)
        - TOP/BOTTOM have mitered corners following FRONT/BACK trim values
        """
        self.assertXMLOutput(
            '''<root>
                <panel thickness="0.1"/>
                <panel thickness="0.1" rotate="90"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.05|1.05,0,0.05|1.05,1,0.05|0,1,0.05"/>
                    <r id="back" pts="0.95,0,-0.05|0,0,-0.05|0,1,-0.05|0.95,1,-0.05"/>
                    <r id="top" pts="0,1,-0.05|0,1,0.05|1.05,1,0.05|0.95,1,-0.05"/>
                    <r id="bottom" pts="0,0,-0.05|0.95,0,-0.05|1.05,0,0.05|0,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                </r>
                <r id="1" pts="1,0,0|1,0,-1|1,1,-1|1,1,0">
                    <r id="front" pts="1.05,0,0.05|1.05,0,-1|1.05,1,-1|1.05,1,0.05"/>
                    <r id="back" pts="0.95,0,-1|0.95,0,-0.05|0.95,1,-0.05|0.95,1,-1"/>
                    <r id="top" pts="0.95,1,-0.05|1.05,1,0.05|1.05,1,-1|0.95,1,-1"/>
                    <r id="bottom" pts="0.95,0,-0.05|0.95,0,-1|1.05,0,-1|1.05,0,0.05"/>
                    <r id="end" pts="1.05,0,-1|0.95,0,-1|0.95,1,-1|1.05,1,-1"/>
                </r>
            </root>'''
        )

    def testThreePanelJointCaps(self):
        """Test that 3-panel joint generates cap polygons to fill the triangular gap.
        
        Three panels meeting at a single joint:
        - Panel 0: along +X from origin
        - Panel 1: rotated 80° from panel 0's end  
        - Panel 2: rotated -130°, attached at panel 0's start (origin)
        
        This creates a Y-shaped intersection at x=1 (end of panel 0, start of panels 1 and 2
        after attachment). The mitered faces leave a triangular gap that must be filled
        with cap polygons.
        
        Expected cap vertices at TOP (y=1):
            [0.976685, 1.0, 0.05]
            [0.958045, 1.0, -0.05]
            [1.05329, 1.0, -0.0142788]
        
        Bottom cap has same x,z coordinates but y=0.
        Top cap winding is reversed so normal faces +Y.
        """
        xml_input = """<root>
            <panel thickness="0.1"/>
            <panel thickness="0.1" rotate="80"/>
            <panel thickness="0.1" rotate="-130" attach-to="0.0"/>
        </root>"""

        expected_xml = """<root>
            <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                <r id="front" pts="0,0,0.05|0.976685,0,0.05|0.976685,1,0.05|0,1,0.05"/>
                <r id="back" pts="0.958045,0,-0.05|0,0,-0.05|0,1,-0.05|0.958045,1,-0.05"/>
                <r id="top" pts="0,1,-0.05|0,1,0.05|0.976685,1,0.05|0.958045,1,-0.05"/>
                <r id="bottom" pts="0,0,-0.05|0.958045,0,-0.05|0.976685,0,0.05|0,0,0.05"/>
                <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                <r id="cap-top" type="polygon" pts="0.976685,1,0.05|1.05329,1,-0.0142788|0.958045,1,-0.05"/>
                <r id="cap-bottom" type="polygon" pts="0.958045,0,-0.05|1.05329,0,-0.0142788|0.976685,0,0.05"/>
            </r>
            <r id="1" pts="1,0,0|1.17365,0,-0.984808|1.17365,1,-0.984808|1,1,0">
                <r id="front" pts="1.05329,0,-0.0142788|1.22289,0,-0.976125|1.22289,1,-0.976125|1.05329,1,-0.0142788"/>
                <r id="back" pts="1.12441,0,-0.99349|0.958045,0,-0.05|0.958045,1,-0.05|1.12441,1,-0.99349"/>
                <r id="top" pts="0.958045,1,-0.05|1.05329,1,-0.0142788|1.22289,1,-0.976125|1.12441,1,-0.99349"/>
                <r id="bottom" pts="0.958045,0,-0.05|1.12441,0,-0.99349|1.22289,0,-0.976125|1.05329,0,-0.0142788"/>
                <r id="end" pts="1.22289,0,-0.976125|1.12441,0,-0.99349|1.12441,1,-0.99349|1.22289,1,-0.976125"/>
            </r>
            <r id="2" pts="1,0,0|1.64279,0,0.766044|1.64279,1,0.766044|1,1,0">
                <r id="front" pts="0.976685,0,0.05|1.60449,0,0.798184|1.60449,1,0.798184|0.976685,1,0.05"/>
                <r id="back" pts="1.68109,0,0.733905|1.05329,0,-0.0142788|1.05329,1,-0.0142788|1.68109,1,0.733905"/>
                <r id="top" pts="1.05329,1,-0.0142788|0.976685,1,0.05|1.60449,1,0.798184|1.68109,1,0.733905"/>
                <r id="bottom" pts="1.05329,0,-0.0142788|1.68109,0,0.733905|1.60449,0,0.798184|0.976685,0,0.05"/>
                <r id="end" pts="1.60449,0,0.798184|1.68109,0,0.733905|1.68109,1,0.733905|1.60449,1,0.798184"/>
            </r>
        </root>"""

        self.assertXMLOutput(xml_input, expected_xml)

    def testSixPanelJointCaps(self):
        """Test that 6-panel joint generates hexagonal cap polygons.
        
        Six panels meeting at a single joint at x=1:
        - Panel 0: along +X from origin
        - Panel 1: rotated 60° from panel 0's end
        - Panels 2-5: all attached at panel 0's start (origin) with various rotations
        
        This creates a 6-way star intersection. The mitered faces leave a hexagonal 
        gap that must be filled with 6-vertex cap polygons.
        """
        xml_input = """<root>
            <panel thickness="0.1"/>
            <panel thickness="0.1" rotate="60"/>
            <panel thickness="0.1" rotate="220" attach-to="0.0"/>
            <panel thickness="0.1" rotate="-120" attach-to="0.0"/>
            <panel thickness="0.1" rotate="50" attach-to="0.0"/>
            <panel thickness="0.1" rotate="50" attach-to="0.0"/>
        </root>"""

        expected_xml = """<root>
            <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                <r id="front" pts="0,0,0.05|0.813397,0,0.05|0.813397,1,0.05|0,1,0.05"/>
                <r id="back" pts="0.716436,0,-0.05|0,0,-0.05|0,1,-0.05|0.716436,1,-0.05"/>
                <r id="top" pts="0,1,-0.05|0,1,0.05|0.813397,1,0.05|0.716436,1,-0.05"/>
                <r id="bottom" pts="0,0,-0.05|0.716436,0,-0.05|0.813397,0,0.05|0,0,0.05"/>
                <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
            </r>
            <r id="1" pts="1,0,0|1.5,0,-0.866025|1.5,1,-0.866025|1,1,0">
                <r id="front" pts="1.0524,0,0.00923963|1.5433,0,-0.841025|1.5433,1,-0.841025|1.0524,1,0.00923963"/>
                <r id="back" pts="1.4567,0,-0.891025|0.977676,0,-0.0613341|0.977676,1,-0.0613341|1.4567,1,-0.891025"/>
                <r id="top" pts="0.977676,1,-0.0613341|1.0524,1,0.00923963|1.5433,1,-0.841025|1.4567,1,-0.891025"/>
                <r id="bottom" pts="0.977676,0,-0.0613341|1.4567,0,-0.891025|1.5433,0,-0.841025|1.0524,0,0.00923963"/>
                <r id="end" pts="1.5433,0,-0.841025|1.4567,0,-0.891025|1.4567,1,-0.891025|1.5433,1,-0.841025"/>
            </r>
            <r id="2" pts="1,0,0|1.17365,0,0.984808|1.17365,1,0.984808|1,1,0">
                <r id="front" pts="1,0,0.287939|1.12441,0,0.99349|1.12441,1,0.99349|1,1,0.287939"/>
                <r id="back" pts="1.22289,0,0.976125|1.0524,0,0.00923963|1.0524,1,0.00923963|1.22289,1,0.976125"/>
                <r id="top" pts="1.0524,1,0.00923963|1,1,0.287939|1.12441,1,0.99349|1.22289,1,0.976125"/>
                <r id="bottom" pts="1.0524,0,0.00923963|1.22289,0,0.976125|1.12441,0,0.99349|1,0,0.287939"/>
                <r id="end" pts="1.12441,0,0.99349|1.22289,0,0.976125|1.22289,1,0.976125|1.12441,1,0.99349"/>
            </r>
            <r id="3" pts="1,0,0|0.0603074,0,-0.34202|0.0603074,1,-0.34202|1,1,0">
                <r id="front" pts="0.977676,0,-0.0613341|0.0774084,0,-0.389005|0.0774084,1,-0.389005|0.977676,1,-0.0613341"/>
                <r id="back" pts="0.0432064,0,-0.295036|0.716436,0,-0.05|0.716436,1,-0.05|0.0432064,1,-0.295036"/>
                <r id="top" pts="0.716436,1,-0.05|0.977676,1,-0.0613341|0.0774084,1,-0.389005|0.0432064,1,-0.295036"/>
                <r id="bottom" pts="0.716436,0,-0.05|0.0432064,0,-0.295036|0.0774084,0,-0.389005|0.977676,0,-0.0613341"/>
                <r id="end" pts="0.0774084,0,-0.389005|0.0432064,0,-0.295036|0.0432064,1,-0.295036|0.0774084,1,-0.389005"/>
                <r id="cap-top" type="polygon" pts="0.716436,1,-0.05|0.813397,1,0.05|0.93214,1,0.0969139|1,1,0.287939|1.0524,1,0.00923963|0.977676,1,-0.0613341"/>
                <r id="cap-bottom" type="polygon" pts="0.977676,0,-0.0613341|1.0524,0,0.00923963|1,0,0.287939|0.93214,0,0.0969139|0.813397,0,0.05|0.716436,0,-0.05"/>
            </r>
            <r id="4" pts="1,0,0|0.133975,0,0.5|0.133975,1,0.5|1,1,0">
                <r id="front" pts="0.813397,0,0.05|0.108975,0,0.456699|0.108975,1,0.456699|0.813397,1,0.05"/>
                <r id="back" pts="0.158975,0,0.543301|0.93214,0,0.0969139|0.93214,1,0.0969139|0.158975,1,0.543301"/>
                <r id="top" pts="0.93214,1,0.0969139|0.813397,1,0.05|0.108975,1,0.456699|0.158975,1,0.543301"/>
                <r id="bottom" pts="0.93214,0,0.0969139|0.158975,0,0.543301|0.108975,0,0.456699|0.813397,0,0.05"/>
                <r id="end" pts="0.108975,0,0.456699|0.158975,0,0.543301|0.158975,1,0.543301|0.108975,1,0.456699"/>
            </r>
            <r id="5" pts="1,0,0|0.826352,0,0.984808|0.826352,1,0.984808|1,1,0">
                <r id="front" pts="0.93214,0,0.0969139|0.777111,0,0.976125|0.777111,1,0.976125|0.93214,1,0.0969139"/>
                <r id="back" pts="0.875592,0,0.99349|1,0,0.287939|1,1,0.287939|0.875592,1,0.99349"/>
                <r id="top" pts="1,1,0.287939|0.93214,1,0.0969139|0.777111,1,0.976125|0.875592,1,0.99349"/>
                <r id="bottom" pts="1,0,0.287939|0.875592,0,0.99349|0.777111,0,0.976125|0.93214,0,0.0969139"/>
                <r id="end" pts="0.777111,0,0.976125|0.875592,0,0.99349|0.875592,1,0.99349|0.777111,1,0.976125"/>
            </r>
        </root>"""

        self.assertXMLOutput(xml_input, expected_xml)

    def testCrossingWithJoints(self):
        """Test a crossing (two diagonal panels) combined with a square frame of joints.
        
        Creates a square frame with 4 panels (0-3) connected by 90° joints, then adds
        two diagonal panels (4-5) crossing through the center. This tests the interaction
        between joint mitering and crossing panel face splits.
        
        Panel layout:
        - Panels 0-3: Square frame with 90° corners
        - Panel 4: Diagonal from panel 1's end to cross panel 2 
        - Panel 5: Diagonal from panel 0's end to cross panels, attached to panel 0
        
        Features tested:
        - Joint caps at 3-panel corners (panels 0,1,3 and panels 2,3,0)
        - Face splits where diagonal panels cross through each other
        - Combined joint mitering and crossing face trimming
        """
        xml_input = """<root>
            <panel thickness="0.1"/>
            <panel thickness="0.1" rotate="90"/>
            <panel thickness="0.1" rotate="90"/>
            <panel thickness="0.1" rotate="90"/>
            
            <panel thickness="0.1" anchor-id="1" anchor-to="1.0"/>
            <panel thickness="0.1" attach-id="0" anchor-id="2" anchor-to="1.0"/>
        </root>"""

        expected_xml = """<root>
            <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                <r id="front" pts="-0.05,0,0.05|1.05,0,0.05|1.05,1,0.05|-0.05,1,0.05"/>
                <r id="back" pts="0.879289,0,-0.05|0.120711,0,-0.05|0.120711,1,-0.05|0.879289,1,-0.05"/>
                <r id="top" pts="0.120711,1,-0.05|-0.05,1,0.05|1.05,1,0.05|0.879289,1,-0.05"/>
                <r id="bottom" pts="0.120711,0,-0.05|0.879289,0,-0.05|1.05,0,0.05|-0.05,0,0.05"/>
                <r id="cap-top" type="polygon" pts="0.120711,1,-0.05|0.05,1,-0.120711|-0.05,1,0.05"/>
                <r id="cap-bottom" type="polygon" pts="-0.05,0,0.05|0.05,0,-0.120711|0.120711,0,-0.05"/>
            </r>
            <r id="1" pts="1,0,0|1,0,-1|1,1,-1|1,1,0">
                <r id="front" pts="1.05,0,0.05|1.05,0,-1.05|1.05,1,-1.05|1.05,1,0.05"/>
                <r id="back" pts="0.95,0,-0.879289|0.95,0,-0.120711|0.95,1,-0.120711|0.95,1,-0.879289"/>
                <r id="top" pts="0.95,1,-0.120711|1.05,1,0.05|1.05,1,-1.05|0.95,1,-0.879289"/>
                <r id="bottom" pts="0.95,0,-0.120711|0.95,0,-0.879289|1.05,0,-1.05|1.05,0,0.05"/>
            </r>
            <r id="2" pts="1,0,-1|0,0,-1|0,1,-1|1,1,-1">
                <r id="front" pts="1.05,0,-1.05|-0.05,0,-1.05|-0.05,1,-1.05|1.05,1,-1.05"/>
                <r id="back" pts="0.120711,0,-0.95|0.879289,0,-0.95|0.879289,1,-0.95|0.120711,1,-0.95"/>
                <r id="top" pts="0.879289,1,-0.95|1.05,1,-1.05|-0.05,1,-1.05|0.120711,1,-0.95"/>
                <r id="bottom" pts="0.879289,0,-0.95|0.120711,0,-0.95|-0.05,0,-1.05|1.05,0,-1.05"/>
                <r id="cap-top" type="polygon" pts="0.879289,1,-0.95|0.95,1,-0.879289|1.05,1,-1.05"/>
                <r id="cap-bottom" type="polygon" pts="1.05,0,-1.05|0.95,0,-0.879289|0.879289,0,-0.95"/>
            </r>
            <r id="3" pts="0,0,-1|0,0,0|0,1,0|0,1,-1">
                <r id="front" pts="-0.05,0,-1.05|-0.05,0,0.05|-0.05,1,0.05|-0.05,1,-1.05"/>
                <r id="back" pts="0.05,0,-0.120711|0.05,0,-0.879289|0.05,1,-0.879289|0.05,1,-0.120711"/>
                <r id="top" pts="0.05,1,-0.879289|-0.05,1,-1.05|-0.05,1,0.05|0.05,1,-0.120711"/>
                <r id="bottom" pts="0.05,0,-0.879289|0.05,0,-0.120711|-0.05,0,0.05|-0.05,0,-1.05"/>
                <r id="cap-top" type="polygon" pts="0.05,1,-0.879289|0.120711,1,-0.95|-0.05,1,-1.05"/>
                <r id="cap-bottom" type="polygon" pts="-0.05,0,-1.05|0.120711,0,-0.95|0.05,0,-0.879289"/>
            </r>
            <r id="4" pts="0,0,0|1,0,-1|1,1,-1|0,1,0">
                <r id="front-0" pts="0.120711,0,-0.05|0.5,0,-0.429289|0.5,1,-0.429289|0.120711,1,-0.05"/>
                <r id="front-1" pts="0.570711,0,-0.5|0.95,0,-0.879289|0.95,1,-0.879289|0.570711,1,-0.5"/>
                <r id="back-0" pts="0.429289,0,-0.5|0.05,0,-0.120711|0.05,1,-0.120711|0.429289,1,-0.5"/>
                <r id="back-1" pts="0.879289,0,-0.95|0.5,0,-0.570711|0.5,1,-0.570711|0.879289,1,-0.95"/>
                <r id="top-0" pts="0.05,1,-0.120711|0.120711,1,-0.05|0.5,1,-0.429289|0.429289,1,-0.5"/>
                <r id="top-1" pts="0.5,1,-0.570711|0.570711,1,-0.5|0.95,1,-0.879289|0.879289,1,-0.95"/>
                <r id="bottom-0" pts="0.05,0,-0.120711|0.429289,0,-0.5|0.5,0,-0.429289|0.120711,0,-0.05"/>
                <r id="bottom-1" pts="0.5,0,-0.570711|0.879289,0,-0.95|0.95,0,-0.879289|0.570711,0,-0.5"/>
                <r id="crossing-cap-top" type="polygon" pts="0.429289,1,-0.5|0.5,1,-0.429289|0.570711,1,-0.5|0.5,1,-0.570711"/>
                <r id="crossing-cap-bottom" type="polygon" pts="0.5,0,-0.570711|0.570711,0,-0.5|0.5,0,-0.429289|0.429289,0,-0.5"/>
            </r>
            <r id="5" pts="1,0,0|0,0,-1|0,1,-1|1,1,0">
                <r id="front-0" pts="0.95,0,-0.120711|0.570711,0,-0.5|0.570711,1,-0.5|0.95,1,-0.120711"/>
                <r id="front-1" pts="0.5,0,-0.570711|0.120711,0,-0.95|0.120711,1,-0.95|0.5,1,-0.570711"/>
                <r id="back-0" pts="0.5,0,-0.429289|0.879289,0,-0.05|0.879289,1,-0.05|0.5,1,-0.429289"/>
                <r id="back-1" pts="0.05,0,-0.879289|0.429289,0,-0.5|0.429289,1,-0.5|0.05,1,-0.879289"/>
                <r id="top-0" pts="0.879289,1,-0.05|0.95,1,-0.120711|0.570711,1,-0.5|0.5,1,-0.429289"/>
                <r id="top-1" pts="0.429289,1,-0.5|0.5,1,-0.570711|0.120711,1,-0.95|0.05,1,-0.879289"/>
                <r id="bottom-0" pts="0.879289,0,-0.05|0.5,0,-0.429289|0.570711,0,-0.5|0.95,0,-0.120711"/>
                <r id="bottom-1" pts="0.429289,0,-0.5|0.05,0,-0.879289|0.120711,0,-0.95|0.5,0,-0.570711"/>
                <r id="cap-top" type="polygon" pts="0.879289,1,-0.05|1.05,1,0.05|0.95,1,-0.120711"/>
                <r id="cap-bottom" type="polygon" pts="0.95,0,-0.120711|1.05,0,0.05|0.879289,0,-0.05"/>
            </r>
        </root>"""

        self.assertXMLOutput(xml_input, expected_xml)


class TJunctionTests(BaseIntegrationTest):
    """Tests for T-junction panel intersections."""
    
    def testBasicTJunction(self):
        """Test a basic T-junction where one panel meets another at 90 degrees.
        
        Panel 1 is rotated 90 degrees and meets the back face of Panel 0 at x=0.5.
        The back face of Panel 0 should be split into two regions: back-0 and back-1.
        A gap of 0.1 (the thickness of panel 1) is created at the intersection point,
        so back-0 ends at 0.45 and back-1 starts at 0.55.
        
        Panel 1's faces are trimmed to meet panel 0's back face surface (z=-0.05).
        The START face is omitted since it's interior geometry (closed by panel 0's back face).
        """
        self.assertXMLOutput(
            '''<root>
                <panel thickness="0.1"/>
                <panel thickness="0.1" rotate="90" offset="0,0,-0.5"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.05|1,0,0.05|1,1,0.05|0,1,0.05"/>
                    <r id="back-0" pts="0.45,0,-0.05|0,0,-0.05|0,1,-0.05|0.45,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.55,0,-0.05|0.55,1,-0.05|1,1,-0.05"/>
                    <r id="top" pts="0,1,-0.05|0,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom" pts="0,0,-0.05|1,0,-0.05|1,0,0.05|0,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                </r>
                <r id="1" pts="0.5,0,0|0.5,0,-1|0.5,1,-1|0.5,1,0">
                    <r id="front" pts="0.55,0,-0.05|0.55,0,-1|0.55,1,-1|0.55,1,-0.05"/>
                    <r id="back" pts="0.45,0,-1|0.45,0,-0.05|0.45,1,-0.05|0.45,1,-1"/>
                    <r id="top" pts="0.45,1,-0.05|0.55,1,-0.05|0.55,1,-1|0.45,1,-1"/>
                    <r id="bottom" pts="0.45,0,-0.05|0.45,0,-1|0.55,0,-1|0.55,0,-0.05"/>
                    <r id="end" pts="0.55,0,-1|0.45,0,-1|0.45,1,-1|0.55,1,-1"/>
                </r>
            </root>'''
        )

    def testAngledTJunction(self):
        """Test a T-junction where panel 1 is rotated 45 degrees.
        
        Panel 0 has attach-point="0.5", so panel 1 attaches at x=0.5.
        Panel 1 is rotated 45 degrees and meets panel 0's back face.
        The back face of Panel 0 should be split with a gap for the intersecting panel.
        
        At 45 degrees, the gap is wider than for a perpendicular panel (thickness * sqrt(2)).
        The gap is also NOT centered at x=0.5 on the back face because the back face is
        at z=-0.05, and the angled panel's center at z=-0.05 is at x=0.55 (shifted due to angle).
        Gap edges are at approximately 0.479 and 0.621.
        
        Panel 1's faces are trimmed per-face to meet panel 0's back face surface:
        - FRONT face trims more (acute angle) to reach z=-0.05 at x=0.621 (gap end)
        - BACK face trims less (obtuse angle) to reach z=-0.05 at x=0.479 (gap start)
        - TOP/BOTTOM use per-corner trims matching FRONT/BACK
        - START face is omitted (interior geometry, closed by panel 0's back face)
        """
        self.assertXMLOutput(
            '''<root>
                <panel attach-point="0.5" thickness="0.1"/>
                <panel rotate="45" thickness="0.1"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.05|1,0,0.05|1,1,0.05|0,1,0.05"/>
                    <r id="back-0" pts="0.479289,0,-0.05|0,0,-0.05|0,1,-0.05|0.479289,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.620711,0,-0.05|0.620711,1,-0.05|1,1,-0.05"/>
                    <r id="top" pts="0,1,-0.05|0,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom" pts="0,0,-0.05|1,0,-0.05|1,0,0.05|0,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                </r>
                <r id="1" pts="0.5,0,0|1.207107,0,-0.707107|1.207107,1,-0.707107|0.5,1,0">
                    <r id="front" pts="0.620711,0,-0.05|1.242462,0,-0.671751|1.242462,1,-0.671751|0.620711,1,-0.05"/>
                    <r id="back" pts="1.171751,0,-0.742462|0.479289,0,-0.05|0.479289,1,-0.05|1.171751,1,-0.742462"/>
                    <r id="top" pts="0.479289,1,-0.05|0.620711,1,-0.05|1.242462,1,-0.671751|1.171751,1,-0.742462"/>
                    <r id="bottom" pts="0.479289,0,-0.05|1.171751,0,-0.742462|1.242462,0,-0.671751|0.620711,0,-0.05"/>
                    <r id="end" pts="1.242462,0,-0.671751|1.171751,0,-0.742462|1.171751,1,-0.742462|1.242462,1,-0.671751"/>
                </r>
            </root>'''
        )

    def testTJunctionAt30Degrees(self):
        """Test a T-junction at 30 degrees (acute angle, less than 45).
        
        At shallow angles, the gap in the intersected panel becomes wider
        and the trim calculations must correctly handle the geometry.
        """
        self.assertXMLOutput(
            '''<root>
                <panel attach-point="0.5" thickness="0.1"/>
                <panel rotate="30" thickness="0.1"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.05|1,0,0.05|1,1,0.05|0,1,0.05"/>
                    <r id="back-0" pts="0.486603,0,-0.05|0,0,-0.05|0,1,-0.05|0.486603,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.686603,0,-0.05|0.686603,1,-0.05|1,1,-0.05"/>
                    <r id="top" pts="0,1,-0.05|0,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom" pts="0,0,-0.05|1,0,-0.05|1,0,0.05|0,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                </r>
                <r id="1" pts="0.5,0,0|1.36603,0,-0.5|1.36603,1,-0.5|0.5,1,0">
                    <r id="front" pts="0.686603,0,-0.05|1.39103,0,-0.456699|1.39103,1,-0.456699|0.686603,1,-0.05"/>
                    <r id="back" pts="1.34103,0,-0.543301|0.486603,0,-0.05|0.486603,1,-0.05|1.34103,1,-0.543301"/>
                    <r id="top" pts="0.486603,1,-0.05|0.686603,1,-0.05|1.39103,1,-0.456699|1.34103,1,-0.543301"/>
                    <r id="bottom" pts="0.486603,0,-0.05|1.34103,0,-0.543301|1.39103,0,-0.456699|0.686603,0,-0.05"/>
                    <r id="end" pts="1.39103,0,-0.456699|1.34103,0,-0.543301|1.34103,1,-0.543301|1.39103,1,-0.456699"/>
                </r>
            </root>'''
        )

    def testTJunctionAt44Degrees(self):
        """Test a T-junction at 44 degrees (just under 45).
        
        This tests the edge case where the approach direction has a larger
        dot product with the END face than the BACK face. The geometry builder
        must correctly identify BACK as the intersection target.
        """
        self.assertXMLOutput(
            '''<root>
                <panel attach-point="0.5" thickness="0.1"/>
                <panel rotate="44" thickness="0.1"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.05|1,0,0.05|1,1,0.05|0,1,0.05"/>
                    <r id="back-0" pts="0.479799,0,-0.05|0,0,-0.05|0,1,-0.05|0.479799,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.623754,0,-0.05|0.623754,1,-0.05|1,1,-0.05"/>
                    <r id="top" pts="0,1,-0.05|0,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom" pts="0,0,-0.05|1,0,-0.05|1,0,0.05|0,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                </r>
                <r id="1" pts="0.5,0,0|1.21934,0,-0.694658|1.21934,1,-0.694658|0.5,1,0">
                    <r id="front" pts="0.623754,0,-0.05|1.25407,0,-0.658691|1.25407,1,-0.658691|0.623754,1,-0.05"/>
                    <r id="back" pts="1.18461,0,-0.730625|0.479799,0,-0.05|0.479799,1,-0.05|1.18461,1,-0.730625"/>
                    <r id="top" pts="0.479799,1,-0.05|0.623754,1,-0.05|1.25407,1,-0.658691|1.18461,1,-0.730625"/>
                    <r id="bottom" pts="0.479799,0,-0.05|1.18461,0,-0.730625|1.25407,0,-0.658691|0.623754,0,-0.05"/>
                    <r id="end" pts="1.25407,0,-0.658691|1.18461,0,-0.730625|1.18461,1,-0.730625|1.25407,1,-0.658691"/>
                </r>
            </root>'''
        )

    def testTJunctionAt135Degrees(self):
        """Test a T-junction at 135 degrees (obtuse angle).
        
        At obtuse angles, the panel extends in the opposite X direction.
        The intersection logic must handle this correctly.
        """
        self.assertXMLOutput(
            '''<root>
                <panel attach-point="0.5" thickness="0.1"/>
                <panel rotate="135" thickness="0.1"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.05|1,0,0.05|1,1,0.05|0,1,0.05"/>
                    <r id="back-0" pts="0.379289,0,-0.05|0,0,-0.05|0,1,-0.05|0.379289,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.520711,0,-0.05|0.520711,1,-0.05|1,1,-0.05"/>
                    <r id="top" pts="0,1,-0.05|0,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom" pts="0,0,-0.05|1,0,-0.05|1,0,0.05|0,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                </r>
                <r id="1" pts="0.5,0,0|-0.207107,0,-0.707107|-0.207107,1,-0.707107|0.5,1,0">
                    <r id="front" pts="0.520711,0,-0.05|-0.171751,0,-0.742462|-0.171751,1,-0.742462|0.520711,1,-0.05"/>
                    <r id="back" pts="-0.242462,0,-0.671751|0.379289,0,-0.05|0.379289,1,-0.05|-0.242462,1,-0.671751"/>
                    <r id="top" pts="0.379289,1,-0.05|0.520711,1,-0.05|-0.171751,1,-0.742462|-0.242462,1,-0.671751"/>
                    <r id="bottom" pts="0.379289,0,-0.05|-0.242462,0,-0.671751|-0.171751,0,-0.742462|0.520711,0,-0.05"/>
                    <r id="end" pts="-0.171751,0,-0.742462|-0.242462,0,-0.671751|-0.242462,1,-0.671751|-0.171751,1,-0.742462"/>
                </r>
            </root>'''
        )

    def testTJunctionAt150Degrees(self):
        """Test a T-junction at 150 degrees (steep obtuse angle).
        
        Similar to 30 degrees but in the opposite direction. Tests that
        the geometry builder handles extreme obtuse angles correctly.
        """
        self.assertXMLOutput(
            '''<root>
                <panel attach-point="0.5" thickness="0.1"/>
                <panel rotate="150" thickness="0.1"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front" pts="0,0,0.05|1,0,0.05|1,1,0.05|0,1,0.05"/>
                    <r id="back-0" pts="0.313397,0,-0.05|0,0,-0.05|0,1,-0.05|0.313397,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.513397,0,-0.05|0.513397,1,-0.05|1,1,-0.05"/>
                    <r id="top" pts="0,1,-0.05|0,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom" pts="0,0,-0.05|1,0,-0.05|1,0,0.05|0,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                </r>
                <r id="1" pts="0.5,0,0|-0.366025,0,-0.5|-0.366025,1,-0.5|0.5,1,0">
                    <r id="front" pts="0.513397,0,-0.05|-0.341025,0,-0.543301|-0.341025,1,-0.543301|0.513397,1,-0.05"/>
                    <r id="back" pts="-0.391025,0,-0.456699|0.313397,0,-0.05|0.313397,1,-0.05|-0.391025,1,-0.456699"/>
                    <r id="top" pts="0.313397,1,-0.05|0.513397,1,-0.05|-0.341025,1,-0.543301|-0.391025,1,-0.456699"/>
                    <r id="bottom" pts="0.313397,0,-0.05|-0.391025,0,-0.456699|-0.341025,0,-0.543301|0.513397,0,-0.05"/>
                    <r id="end" pts="-0.341025,0,-0.543301|-0.391025,0,-0.456699|-0.391025,1,-0.456699|-0.341025,1,-0.543301"/>
                </r>
            </root>'''
        )

    def testOrthogonalTJunctions(self):
        """Test multiple orthogonal T-junctions on a single panel.
        
        Panel 0 is a 10-unit panel with two perpendicular panels attached:
        - Panel 1 attaches at 75% (7.5 units) rotated +90 degrees (into -Z)
        - Panel 2 attaches at 0% (start) rotated -90 degrees (into +Z)
        
        Panel 0's FRONT face is NOT SPLIT because panel 1's T-junction approaches
        the BACK face, not front. Panel 2 attaches at the endpoint which only affects
        the start trim, not face splitting.
        Panel 0's BACK face IS SPLIT with a GAP for panel 1's thickness (0.5 units).
        Panel 0 loses its start cap (closed by panel 2's back face).
        
        The front face is a single unsplit region.
        The back face has a gap from x=7.25 to x=7.75.
        """
        self.assertXMLOutput(
            '''<root>
                <panel size="10" thickness="0.5" rotate="0"/>
                <panel attach-id="0" attach-to="0.75" rotate="90" size="4" thickness="0.5"/>
                <panel attach-id="0" attach-to="0" rotate="-90" size="4" thickness="0.5"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|10,0,0|10,1,0|0,1,0">
                    <r id="front" pts="0.25,0,0.25|10,0,0.25|10,1,0.25|0.25,1,0.25"/>
                    <r id="back-0" pts="7.25,0,-0.25|-0.25,0,-0.25|-0.25,1,-0.25|7.25,1,-0.25"/>
                    <r id="back-1" pts="10,0,-0.25|7.75,0,-0.25|7.75,1,-0.25|10,1,-0.25"/>
                    <r id="top" pts="-0.25,1,-0.25|0.25,1,0.25|10,1,0.25|10,1,-0.25"/>
                    <r id="bottom" pts="-0.25,0,-0.25|10,0,-0.25|10,0,0.25|0.25,0,0.25"/>
                    <r id="end" pts="10,0,0.25|10,0,-0.25|10,1,-0.25|10,1,0.25"/>
                </r>
                <r id="1" pts="7.5,0,0|7.5,0,-4|7.5,1,-4|7.5,1,0">
                    <r id="front" pts="7.75,0,-0.25|7.75,0,-4|7.75,1,-4|7.75,1,-0.25"/>
                    <r id="back" pts="7.25,0,-4|7.25,0,-0.25|7.25,1,-0.25|7.25,1,-4"/>
                    <r id="top" pts="7.25,1,-0.25|7.75,1,-0.25|7.75,1,-4|7.25,1,-4"/>
                    <r id="bottom" pts="7.25,0,-0.25|7.25,0,-4|7.75,0,-4|7.75,0,-0.25"/>
                    <r id="end" pts="7.75,0,-4|7.25,0,-4|7.25,1,-4|7.75,1,-4"/>
                </r>
                <r id="2" pts="0,0,0|2.44929e-16,0,4|2.44929e-16,1,4|0,1,0">
                    <r id="front" pts="-0.25,0,-0.25|-0.25,0,4|-0.25,1,4|-0.25,1,-0.25"/>
                    <r id="back" pts="0.25,0,4|0.25,0,0.25|0.25,1,0.25|0.25,1,4"/>
                    <r id="top" pts="0.25,1,0.25|-0.25,1,-0.25|-0.25,1,4|0.25,1,4"/>
                    <r id="bottom" pts="0.25,0,0.25|0.25,0,4|-0.25,0,4|-0.25,0,-0.25"/>
                    <r id="end" pts="-0.25,0,4|0.25,0,4|0.25,1,4|-0.25,1,4"/>
                </r>
            </root>'''
        )


class CrossingTests(BaseIntegrationTest):
    """Tests for crossing panel intersections where both panels are split."""
    
    def testBasicCrossing(self):
        """Test a basic perpendicular crossing where both panels pass through each other.
        
        Panel 0 is horizontal along X axis. Panel 1 is rotated 90 degrees and 
        positioned to cross through Panel 0's center at x=0.5.
        
        All four lengthwise faces (FRONT, BACK, TOP, BOTTOM) are split into two segments
        with a gap at the intersection point. The gap width equals the intersecting
        panel's thickness (0.1), centered at x=0.5 on both panels. Crossing caps fill
        the rectangular gaps in TOP/BOTTOM faces.
        """
        self.assertXMLOutput(
            '''<root>
                <panel thickness="0.1"/>
                <panel rotate="90" thickness="0.1" offset="-0.5,0,-0.5"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front-0" pts="0,0,0.05|0.45,0,0.05|0.45,1,0.05|0,1,0.05"/>
                    <r id="front-1" pts="0.55,0,0.05|1,0,0.05|1,1,0.05|0.55,1,0.05"/>
                    <r id="back-0" pts="0.45,0,-0.05|0,0,-0.05|0,1,-0.05|0.45,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.55,0,-0.05|0.55,1,-0.05|1,1,-0.05"/>
                    <r id="top-0" pts="0,1,-0.05|0,1,0.05|0.45,1,0.05|0.45,1,-0.05"/>
                    <r id="top-1" pts="0.55,1,-0.05|0.55,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom-0" pts="0,0,-0.05|0.45,0,-0.05|0.45,0,0.05|0,0,0.05"/>
                    <r id="bottom-1" pts="0.55,0,-0.05|1,0,-0.05|1,0,0.05|0.55,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                    <r id="crossing-cap-top" type="polygon" pts="0.45,1,0.05|0.55,1,0.05|0.55,1,-0.05|0.45,1,-0.05"/>
                    <r id="crossing-cap-bottom" type="polygon" pts="0.45,0,-0.05|0.55,0,-0.05|0.55,0,0.05|0.45,0,0.05"/>
                </r>
                <r id="1" pts="0.5,0,0.5|0.5,0,-0.5|0.5,1,-0.5|0.5,1,0.5">
                    <r id="front-0" pts="0.55,0,0.5|0.55,0,0.05|0.55,1,0.05|0.55,1,0.5"/>
                    <r id="front-1" pts="0.55,0,-0.05|0.55,0,-0.5|0.55,1,-0.5|0.55,1,-0.05"/>
                    <r id="back-0" pts="0.45,0,0.05|0.45,0,0.5|0.45,1,0.5|0.45,1,0.05"/>
                    <r id="back-1" pts="0.45,0,-0.5|0.45,0,-0.05|0.45,1,-0.05|0.45,1,-0.5"/>
                    <r id="top-0" pts="0.45,1,0.5|0.55,1,0.5|0.55,1,0.05|0.45,1,0.05"/>
                    <r id="top-1" pts="0.45,1,-0.05|0.55,1,-0.05|0.55,1,-0.5|0.45,1,-0.5"/>
                    <r id="bottom-0" pts="0.45,0,0.5|0.45,0,0.05|0.55,0,0.05|0.55,0,0.5"/>
                    <r id="bottom-1" pts="0.45,0,-0.05|0.45,0,-0.5|0.55,0,-0.5|0.55,0,-0.05"/>
                    <r id="start" pts="0.45,0,0.5|0.55,0,0.5|0.55,1,0.5|0.45,1,0.5"/>
                    <r id="end" pts="0.55,0,-0.5|0.45,0,-0.5|0.45,1,-0.5|0.55,1,-0.5"/>
                </r>
            </root>'''
        )

    def testAngledCrossing(self):
        """Test a crossing where both panels are rotated 45 degrees.
        
        Panel 0 is rotated 45 degrees. Panel 1 is also rotated 45 degrees but
        offset to cross through Panel 0 perpendicularly (90 degree intersection).
        
        All four lengthwise faces (FRONT, BACK, TOP, BOTTOM) are split into two segments
        with gaps at the intersection point. The crossing cap is a parallelogram (4 vertices)
        formed by the intersections of the gap edges from both panels.
        """
        self.assertXMLOutput(
            '''<root>
                <panel rotate="45" thickness="0.1"/>
                <panel rotate="45" thickness="0.1" offset="-0.75,0,-0.25"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|0.707107,0,-0.707107|0.707107,1,-0.707107|0,1,0">
                    <r id="front-0" pts="0.0353553,0,0.0353553|0.407107,0,-0.336396|0.407107,1,-0.336396|0.0353553,1,0.0353553"/>
                    <r id="front-1" pts="0.507107,0,-0.436396|0.742462,0,-0.671751|0.742462,1,-0.671751|0.507107,1,-0.436396"/>
                    <r id="back-0" pts="0.407107,0,-0.477817|-0.0353553,0,-0.0353553|-0.0353553,1,-0.0353553|0.407107,1,-0.477817"/>
                    <r id="back-1" pts="0.671751,0,-0.742462|0.507107,0,-0.577817|0.507107,1,-0.577817|0.671751,1,-0.742462"/>
                    <r id="top-0" pts="-0.0353553,1,-0.0353553|0.0353553,1,0.0353553|0.407107,1,-0.336396|0.407107,1,-0.477817"/>
                    <r id="top-1" pts="0.507107,1,-0.577817|0.507107,1,-0.436396|0.742462,1,-0.671751|0.671751,1,-0.742462"/>
                    <r id="bottom-0" pts="-0.0353553,0,-0.0353553|0.407107,0,-0.477817|0.407107,0,-0.336396|0.0353553,0,0.0353553"/>
                    <r id="bottom-1" pts="0.507107,0,-0.577817|0.671751,0,-0.742462|0.742462,0,-0.671751|0.507107,0,-0.436396"/>
                    <r id="start" pts="-0.0353553,0,-0.0353553|0.0353553,0,0.0353553|0.0353553,1,0.0353553|-0.0353553,1,-0.0353553"/>
                    <r id="end" pts="0.742462,0,-0.671751|0.671751,0,-0.742462|0.671751,1,-0.742462|0.742462,1,-0.671751"/>
                    <r id="crossing-cap-top" type="polygon" pts="0.407107,1,-0.336396|0.507107,1,-0.436396|0.507107,1,-0.577817|0.407107,1,-0.477817"/>
                    <r id="crossing-cap-bottom" type="polygon" pts="0.407107,0,-0.477817|0.507107,0,-0.577817|0.507107,0,-0.436396|0.407107,0,-0.336396"/>
                </r>
                <r id="1" pts="0.457107,0,0.0428932|0.457107,0,-0.957107|0.457107,1,-0.957107|0.457107,1,0.0428932">
                    <r id="front-0" pts="0.507107,0,0.0428932|0.507107,0,-0.436396|0.507107,1,-0.436396|0.507107,1,0.0428932"/>
                    <r id="front-1" pts="0.507107,0,-0.577817|0.507107,0,-0.957107|0.507107,1,-0.957107|0.507107,1,-0.577817"/>
                    <r id="back-0" pts="0.407107,0,-0.336396|0.407107,0,0.0428932|0.407107,1,0.0428932|0.407107,1,-0.336396"/>
                    <r id="back-1" pts="0.407107,0,-0.957107|0.407107,0,-0.477817|0.407107,1,-0.477817|0.407107,1,-0.957107"/>
                    <r id="top-0" pts="0.407107,1,0.0428932|0.507107,1,0.0428932|0.507107,1,-0.436396|0.407107,1,-0.336396"/>
                    <r id="top-1" pts="0.407107,1,-0.477817|0.507107,1,-0.577817|0.507107,1,-0.957107|0.407107,1,-0.957107"/>
                    <r id="bottom-0" pts="0.407107,0,0.0428932|0.407107,0,-0.336396|0.507107,0,-0.436396|0.507107,0,0.0428932"/>
                    <r id="bottom-1" pts="0.407107,0,-0.477817|0.407107,0,-0.957107|0.507107,0,-0.957107|0.507107,0,-0.577817"/>
                    <r id="start" pts="0.407107,0,0.0428932|0.507107,0,0.0428932|0.507107,1,0.0428932|0.407107,1,0.0428932"/>
                    <r id="end" pts="0.507107,0,-0.957107|0.407107,0,-0.957107|0.407107,1,-0.957107|0.507107,1,-0.957107"/>
                </r>
            </root>'''
        )

    def testMultipleCrossings(self):
        """Test a panel crossed by three perpendicular panels.
        
        Panel 0 is horizontal along X. Panels 1, 2, 3 are rotated 90 degrees
        and cross through Panel 0 at different positions (x=0.25, 0.5, 0.75).
        
        All four lengthwise faces (FRONT, BACK, TOP, BOTTOM) are split into 3 segments each,
        with gaps at each intersection. The crossing panels also get split
        where they intersect each other. Crossing caps fill the rectangular gaps
        where panels pass through each other.
        """
        self.assertXMLOutput(
            '''<root>
                <panel thickness="0.1"/>
                <panel rotate="90" thickness="0.1" offset="-0.25,0,-0.25"/>
                <panel rotate="90" thickness="0.1" offset="0,0,-0.5"/>
                <panel rotate="90" thickness="0.1" offset="0.25,0,-0.25"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front-0" pts="0,0,0.05|0.2,0,0.05|0.2,1,0.05|0,1,0.05"/>
                    <r id="front-1" pts="0.3,0,0.05|0.7,0,0.05|0.7,1,0.05|0.3,1,0.05"/>
                    <r id="front-2" pts="0.8,0,0.05|1,0,0.05|1,1,0.05|0.8,1,0.05"/>
                    <r id="back-0" pts="0.2,0,-0.05|0,0,-0.05|0,1,-0.05|0.2,1,-0.05"/>
                    <r id="back-1" pts="0.7,0,-0.05|0.3,0,-0.05|0.3,1,-0.05|0.7,1,-0.05"/>
                    <r id="back-2" pts="1,0,-0.05|0.8,0,-0.05|0.8,1,-0.05|1,1,-0.05"/>
                    <r id="top-0" pts="0,1,-0.05|0,1,0.05|0.2,1,0.05|0.2,1,-0.05"/>
                    <r id="top-1" pts="0.3,1,-0.05|0.3,1,0.05|0.7,1,0.05|0.7,1,-0.05"/>
                    <r id="top-2" pts="0.8,1,-0.05|0.8,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom-0" pts="0,0,-0.05|0.2,0,-0.05|0.2,0,0.05|0,0,0.05"/>
                    <r id="bottom-1" pts="0.3,0,-0.05|0.7,0,-0.05|0.7,0,0.05|0.3,0,0.05"/>
                    <r id="bottom-2" pts="0.8,0,-0.05|1,0,-0.05|1,0,0.05|0.8,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                    <r id="crossing-cap-top" type="polygon" pts="0.7,1,0.05|0.8,1,0.05|0.8,1,-0.05|0.7,1,-0.05"/>
                    <r id="crossing-cap-bottom" type="polygon" pts="0.7,0,-0.05|0.8,0,-0.05|0.8,0,0.05|0.7,0,0.05"/>
                    <r id="crossing-cap-top" type="polygon" pts="0.2,1,0.05|0.3,1,0.05|0.3,1,-0.05|0.2,1,-0.05"/>
                    <r id="crossing-cap-bottom" type="polygon" pts="0.2,0,-0.05|0.3,0,-0.05|0.3,0,0.05|0.2,0,0.05"/>
                </r>
                <r id="1" pts="0.75,0,0.25|0.75,0,-0.75|0.75,1,-0.75|0.75,1,0.25">
                    <r id="front-0" pts="0.8,0,0.25|0.8,0,0.05|0.8,1,0.05|0.8,1,0.25"/>
                    <r id="front-1" pts="0.8,0,-0.05|0.8,0,-0.45|0.8,1,-0.45|0.8,1,-0.05"/>
                    <r id="front-2" pts="0.8,0,-0.55|0.8,0,-0.75|0.8,1,-0.75|0.8,1,-0.55"/>
                    <r id="back-0" pts="0.7,0,0.05|0.7,0,0.25|0.7,1,0.25|0.7,1,0.05"/>
                    <r id="back-1" pts="0.7,0,-0.45|0.7,0,-0.05|0.7,1,-0.05|0.7,1,-0.45"/>
                    <r id="back-2" pts="0.7,0,-0.75|0.7,0,-0.55|0.7,1,-0.55|0.7,1,-0.75"/>
                    <r id="top-0" pts="0.7,1,0.25|0.8,1,0.25|0.8,1,0.05|0.7,1,0.05"/>
                    <r id="top-1" pts="0.7,1,-0.05|0.8,1,-0.05|0.8,1,-0.45|0.7,1,-0.45"/>
                    <r id="top-2" pts="0.7,1,-0.55|0.8,1,-0.55|0.8,1,-0.75|0.7,1,-0.75"/>
                    <r id="bottom-0" pts="0.7,0,0.25|0.7,0,0.05|0.8,0,0.05|0.8,0,0.25"/>
                    <r id="bottom-1" pts="0.7,0,-0.05|0.7,0,-0.45|0.8,0,-0.45|0.8,0,-0.05"/>
                    <r id="bottom-2" pts="0.7,0,-0.55|0.7,0,-0.75|0.8,0,-0.75|0.8,0,-0.55"/>
                    <r id="start" pts="0.7,0,0.25|0.8,0,0.25|0.8,1,0.25|0.7,1,0.25"/>
                    <r id="end" pts="0.8,0,-0.75|0.7,0,-0.75|0.7,1,-0.75|0.8,1,-0.75"/>
                    <r id="crossing-cap-top" type="polygon" pts="0.7,1,-0.45|0.8,1,-0.45|0.8,1,-0.55|0.7,1,-0.55"/>
                    <r id="crossing-cap-bottom" type="polygon" pts="0.7,0,-0.55|0.8,0,-0.55|0.8,0,-0.45|0.7,0,-0.45"/>
                </r>
                <r id="2" pts="1,0,-0.5|0,0,-0.5|0,1,-0.5|1,1,-0.5">
                    <r id="front-0" pts="1,0,-0.55|0.8,0,-0.55|0.8,1,-0.55|1,1,-0.55"/>
                    <r id="front-1" pts="0.7,0,-0.55|0.3,0,-0.55|0.3,1,-0.55|0.7,1,-0.55"/>
                    <r id="front-2" pts="0.2,0,-0.55|0,0,-0.55|0,1,-0.55|0.2,1,-0.55"/>
                    <r id="back-0" pts="0.8,0,-0.45|1,0,-0.45|1,1,-0.45|0.8,1,-0.45"/>
                    <r id="back-1" pts="0.3,0,-0.45|0.7,0,-0.45|0.7,1,-0.45|0.3,1,-0.45"/>
                    <r id="back-2" pts="0,0,-0.45|0.2,0,-0.45|0.2,1,-0.45|0,1,-0.45"/>
                    <r id="top-0" pts="1,1,-0.45|1,1,-0.55|0.8,1,-0.55|0.8,1,-0.45"/>
                    <r id="top-1" pts="0.7,1,-0.45|0.7,1,-0.55|0.3,1,-0.55|0.3,1,-0.45"/>
                    <r id="top-2" pts="0.2,1,-0.45|0.2,1,-0.55|0,1,-0.55|0,1,-0.45"/>
                    <r id="bottom-0" pts="1,0,-0.45|0.8,0,-0.45|0.8,0,-0.55|1,0,-0.55"/>
                    <r id="bottom-1" pts="0.7,0,-0.45|0.3,0,-0.45|0.3,0,-0.55|0.7,0,-0.55"/>
                    <r id="bottom-2" pts="0.2,0,-0.45|0,0,-0.45|0,0,-0.55|0.2,0,-0.55"/>
                    <r id="start" pts="1,0,-0.45|1,0,-0.55|1,1,-0.55|1,1,-0.45"/>
                    <r id="end" pts="0,0,-0.55|0,0,-0.45|0,1,-0.45|0,1,-0.55"/>
                    <r id="crossing-cap-top" type="polygon" pts="0.2,1,-0.45|0.3,1,-0.45|0.3,1,-0.55|0.2,1,-0.55"/>
                    <r id="crossing-cap-bottom" type="polygon" pts="0.2,0,-0.55|0.3,0,-0.55|0.3,0,-0.45|0.2,0,-0.45"/>
                </r>
                <r id="3" pts="0.25,0,-0.75|0.25,0,0.25|0.25,1,0.25|0.25,1,-0.75">
                    <r id="front-0" pts="0.2,0,-0.75|0.2,0,-0.55|0.2,1,-0.55|0.2,1,-0.75"/>
                    <r id="front-1" pts="0.2,0,-0.45|0.2,0,-0.05|0.2,1,-0.05|0.2,1,-0.45"/>
                    <r id="front-2" pts="0.2,0,0.05|0.2,0,0.25|0.2,1,0.25|0.2,1,0.05"/>
                    <r id="back-0" pts="0.3,0,-0.55|0.3,0,-0.75|0.3,1,-0.75|0.3,1,-0.55"/>
                    <r id="back-1" pts="0.3,0,-0.05|0.3,0,-0.45|0.3,1,-0.45|0.3,1,-0.05"/>
                    <r id="back-2" pts="0.3,0,0.25|0.3,0,0.05|0.3,1,0.05|0.3,1,0.25"/>
                    <r id="top-0" pts="0.3,1,-0.75|0.2,1,-0.75|0.2,1,-0.55|0.3,1,-0.55"/>
                    <r id="top-1" pts="0.3,1,-0.45|0.2,1,-0.45|0.2,1,-0.05|0.3,1,-0.05"/>
                    <r id="top-2" pts="0.3,1,0.05|0.2,1,0.05|0.2,1,0.25|0.3,1,0.25"/>
                    <r id="bottom-0" pts="0.3,0,-0.75|0.3,0,-0.55|0.2,0,-0.55|0.2,0,-0.75"/>
                    <r id="bottom-1" pts="0.3,0,-0.45|0.3,0,-0.05|0.2,0,-0.05|0.2,0,-0.45"/>
                    <r id="bottom-2" pts="0.3,0,0.05|0.3,0,0.25|0.2,0,0.25|0.2,0,0.05"/>
                    <r id="start" pts="0.3,0,-0.75|0.2,0,-0.75|0.2,1,-0.75|0.3,1,-0.75"/>
                    <r id="end" pts="0.2,0,0.25|0.3,0,0.25|0.3,1,0.25|0.2,1,0.25"/>
                </r>
            </root>'''
        )

    def testAngledCrossing45Degrees(self):
        """Test a 45-degree angled crossing where panel 1 passes through panel 0.
        
        Panel 0 is horizontal along X axis. Panel 1 is rotated 45 degrees and 
        attached at x=0.5 of panel 0, with offset=-0.5 to pass through rather than join.
        
        The crossing cap vertices form a parallelogram at the intersection.
        The four vertices are at the intersections of:
        - Panel 0's FRONT edge gap (x=0.379289 to 0.520711 at z=0.05)
        - Panel 0's BACK edge gap (x=0.479289 to 0.620711 at z=-0.05)
        - Panel 1's FRONT and BACK edges (at 45 degrees)
        """
        self.assertXMLOutput(
            '''<root>
                <panel thickness="0.1"/>
                <panel thickness="0.1" rotate="45" attach-id="0" attach-to="0.5" offset="-0.5"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r id="front-0" pts="0,0,0.05|0.379289,0,0.05|0.379289,1,0.05|0,1,0.05"/>
                    <r id="front-1" pts="0.520711,0,0.05|1,0,0.05|1,1,0.05|0.520711,1,0.05"/>
                    <r id="back-0" pts="0.479289,0,-0.05|0,0,-0.05|0,1,-0.05|0.479289,1,-0.05"/>
                    <r id="back-1" pts="1,0,-0.05|0.620711,0,-0.05|0.620711,1,-0.05|1,1,-0.05"/>
                    <r id="top-0" pts="0,1,-0.05|0,1,0.05|0.379289,1,0.05|0.479289,1,-0.05"/>
                    <r id="top-1" pts="0.620711,1,-0.05|0.520711,1,0.05|1,1,0.05|1,1,-0.05"/>
                    <r id="bottom-0" pts="0,0,-0.05|0.479289,0,-0.05|0.379289,0,0.05|0,0,0.05"/>
                    <r id="bottom-1" pts="0.620711,0,-0.05|1,0,-0.05|1,0,0.05|0.520711,0,0.05"/>
                    <r id="start" pts="0,0,-0.05|0,0,0.05|0,1,0.05|0,1,-0.05"/>
                    <r id="end" pts="1,0,0.05|1,0,-0.05|1,1,-0.05|1,1,0.05"/>
                    <r id="crossing-cap-top" type="polygon" pts="0.379289,1,0.05|0.520711,1,0.05|0.620711,1,-0.05|0.479289,1,-0.05"/>
                    <r id="crossing-cap-bottom" type="polygon" pts="0.479289,0,-0.05|0.620711,0,-0.05|0.520711,0,0.05|0.379289,0,0.05"/>
                </r>
                <r id="1" pts="0.146447,0,0.353553|0.853553,0,-0.353553|0.853553,1,-0.353553|0.146447,1,0.353553">
                    <r id="front-0" pts="0.181802,0,0.388909|0.520711,0,0.05|0.520711,1,0.05|0.181802,1,0.388909"/>
                    <r id="front-1" pts="0.620711,0,-0.05|0.888909,0,-0.318198|0.888909,1,-0.318198|0.620711,1,-0.05"/>
                    <r id="back-0" pts="0.379289,0,0.05|0.111091,0,0.318198|0.111091,1,0.318198|0.379289,1,0.05"/>
                    <r id="back-1" pts="0.818198,0,-0.388909|0.479289,0,-0.05|0.479289,1,-0.05|0.818198,1,-0.388909"/>
                    <r id="top-0" pts="0.111091,1,0.318198|0.181802,1,0.388909|0.520711,1,0.05|0.379289,1,0.05"/>
                    <r id="top-1" pts="0.479289,1,-0.05|0.620711,1,-0.05|0.888909,1,-0.318198|0.818198,1,-0.388909"/>
                    <r id="bottom-0" pts="0.111091,0,0.318198|0.379289,0,0.05|0.520711,0,0.05|0.181802,0,0.388909"/>
                    <r id="bottom-1" pts="0.479289,0,-0.05|0.818198,0,-0.388909|0.888909,0,-0.318198|0.620711,0,-0.05"/>
                    <r id="start" pts="0.111091,0,0.318198|0.181802,0,0.388909|0.181802,1,0.388909|0.111091,1,0.318198"/>
                    <r id="end" pts="0.888909,0,-0.318198|0.818198,0,-0.388909|0.818198,1,-0.388909|0.888909,1,-0.318198"/>
                </r>
            </root>'''
        )

    def testComplexAngularIntersection(self):
        """Test complex multi-panel intersection with varied angles and interior panel suppression.
        
        Four panels with different sizes and angles meeting at the origin:
        - Panel 0: 8 units along +X with 0.5 thickness
        - Panel 1: 4 units rotated -90° (into -Z) attached at panel 0's start
        - Panel 2: 8 units rotated 90° (into +Z) attached at panel 0's midpoint
        - Panel 3: 12 units rotated -20° attached at panel 0's start (crosses panel 2)
        
        This creates crossing panels with interior segment suppression disabled (!purge-interior-panels).
        Panel 3 crosses through panel 2, creating segmented faces with crossing-cap polygons.
        """
        self.assertXMLOutput(
            '''<root>
                <panel size="8" thickness="0.5" rotate="0" intersection-options="!purge-interior-panels"/>
                <panel attach-id="0" attach-to="0" rotate="-90" size="4" thickness="0.5" intersection-options="!purge-interior-panels"/>
                <panel attach-to="0.5" rotate="90" size="8" thickness="0.5" intersection-options="!purge-interior-panels"/>
                <panel attach-id="0" attach-to="0" rotate="-20" size="12" thickness="0.15" intersection-options="!purge-interior-panels"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|8,0,0|8,1,0|0,1,0">
                    <r type="polygon" id="front" pts="0.906155,0,0.25|8,0,0.25|8,1,0.25|0.906155,1,0.25"/>
                    <r type="polygon" id="back" pts="8,0,-0.25|-0.25,0,-0.25|-0.25,1,-0.25|8,1,-0.25"/>
                    <r type="polygon" id="top" pts="-0.25,1,-0.25|0.906155,1,0.25|8,1,0.25|8,1,-0.25"/>
                    <r type="polygon" id="bottom" pts="-0.25,0,-0.25|8,0,-0.25|8,0,0.25|0.906155,0,0.25"/>
                    <r type="polygon" id="end" pts="8,0,0.25|8,0,-0.25|8,1,-0.25|8,1,0.25"/>
                </r>
                <r id="1" pts="0,0,0|2.44929e-16,0,4|2.44929e-16,1,4|0,1,0">
                    <r type="polygon" id="front" pts="-0.25,0,-0.25|-0.25,0,4|-0.25,1,4|-0.25,1,-0.25"/>
                    <r type="polygon" id="back-0" pts="0.25,0,1.75|0.25,0,0.170806|0.25,1,0.170806|0.25,1,1.75"/>
                    <r type="polygon" id="back-1" pts="0.25,0,4|0.25,0,2.25|0.25,1,2.25|0.25,1,4"/>
                    <r type="polygon" id="top" pts="0.25,1,0.170806|-0.25,1,-0.25|-0.25,1,4|0.25,1,4"/>
                    <r type="polygon" id="bottom" pts="0.25,0,0.170806|0.25,0,4|-0.25,0,4|-0.25,0,-0.25"/>
                    <r type="polygon" id="end" pts="-0.25,0,4|0.25,0,4|0.25,1,4|-0.25,1,4"/>
                    <r type="polygon" id="cap-top" pts="0.25,1,0.170806|0.906155,1,0.25|-0.25,1,-0.25"/>
                    <r type="polygon" id="cap-bottom" pts="-0.25,0,-0.25|0.906155,0,0.25|0.25,0,0.170806"/>
                </r>
                <r id="2" pts="1.22465e-16,0,2|8,0,2|8,1,2|1.22465e-16,1,2">
                    <r type="polygon" id="front-0" pts="0.25,0,2.25|5.96254,0,2.25|5.96254,1,2.25|0.25,1,2.25"/>
                    <r type="polygon" id="front-1" pts="6.40111,0,2.25|8,0,2.25|8,1,2.25|6.40111,1,2.25"/>
                    <r type="polygon" id="back-0" pts="4.5888,0,1.75|0.25,0,1.75|0.25,1,1.75|4.5888,1,1.75"/>
                    <r type="polygon" id="back-1" pts="8,0,1.75|5.02737,0,1.75|5.02737,1,1.75|8,1,1.75"/>
                    <r type="polygon" id="top-0" pts="0.25,1,1.75|0.25,1,2.25|5.96254,1,2.25|4.5888,1,1.75"/>
                    <r type="polygon" id="top-1" pts="5.02737,1,1.75|6.40111,1,2.25|8,1,2.25|8,1,1.75"/>
                    <r type="polygon" id="bottom-0" pts="0.25,0,1.75|4.5888,0,1.75|5.96254,0,2.25|0.25,0,2.25"/>
                    <r type="polygon" id="bottom-1" pts="5.02737,0,1.75|8,0,1.75|8,0,2.25|6.40111,0,2.25"/>
                    <r type="polygon" id="end" pts="8,0,2.25|8,0,1.75|8,1,1.75|8,1,2.25"/>
                </r>
                <r id="3" pts="0,0,0|11.2763,0,4.10424|11.2763,1,4.10424|0,1,0">
                    <r type="polygon" id="front-0" pts="0.25,0,0.170806|4.5888,0,1.75|4.5888,1,1.75|0.25,1,0.170806"/>
                    <r type="polygon" id="front-1" pts="5.96254,0,2.25|11.2507,0,4.17472|11.2507,1,4.17472|5.96254,1,2.25"/>
                    <r type="polygon" id="back-0" pts="5.02737,0,1.75|0.906155,0,0.25|0.906155,1,0.25|5.02737,1,1.75"/>
                    <r type="polygon" id="back-1" pts="11.302,0,4.03376|6.40111,0,2.25|6.40111,1,2.25|11.302,1,4.03376"/>
                    <r type="polygon" id="top-0" pts="0.906155,1,0.25|0.25,1,0.170806|4.5888,1,1.75|5.02737,1,1.75"/>
                    <r type="polygon" id="top-1" pts="6.40111,1,2.25|5.96254,1,2.25|11.2507,1,4.17472|11.302,1,4.03376"/>
                    <r type="polygon" id="bottom-0" pts="0.906155,0,0.25|5.02737,0,1.75|4.5888,0,1.75|0.25,0,0.170806"/>
                    <r type="polygon" id="bottom-1" pts="6.40111,0,2.25|11.302,0,4.03376|11.2507,0,4.17472|5.96254,0,2.25"/>
                    <r type="polygon" id="end" pts="11.2507,0,4.17472|11.302,0,4.03376|11.302,1,4.03376|11.2507,1,4.17472"/>
                    <r type="polygon" id="crossing-cap-top" pts="5.96254,1,2.25|6.40111,1,2.25|5.02737,1,1.75|4.5888,1,1.75"/>
                    <r type="polygon" id="crossing-cap-bottom" pts="4.5888,0,1.75|5.02737,0,1.75|6.40111,0,2.25|5.96254,0,2.25"/>
                </r>
            </root>'''
        )

    def testVerticalOffsetJoint(self):
        """Test joint where panel 2 has vertical offset via pivot.
        
        Panel 0: 1x1 panel at origin
        Panel 1: 1x2 panel rotated 90°, attached with vertical pivot offset
        
        The pivot="0,0.5" places panel 1's pivot at half its height (y=0.5 from its local origin).
        attach-to="1,0.5" attaches at the end of panel 0 at half height.
        This creates a panel that extends from y=-0.5 to y=1.5 (2 units tall, centered at y=0.5).
        """
        self.assertXMLOutput(
            '''<root>
                <panel size="1,1" thickness="0.5"/>
                <panel size="1,2" pivot="0,0.5" attach-to="1,0.5" rotate="90" thickness="0.5"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|1,0,0|1,1,0|0,1,0">
                    <r type="polygon" id="front" pts="0,0,0.25|1,0,0.25|1,1,0.25|0,1,0.25"/>
                    <r type="polygon" id="back" pts="1,0,-0.25|0,0,-0.25|0,1,-0.25|1,1,-0.25"/>
                    <r type="polygon" id="top" pts="0,1,-0.25|0,1,0.25|1,1,0.25|1,1,-0.25"/>
                    <r type="polygon" id="bottom" pts="0,0,-0.25|1,0,-0.25|1,0,0.25|0,0,0.25"/>
                    <r type="polygon" id="start" pts="0,0,-0.25|0,0,0.25|0,1,0.25|0,1,-0.25"/>
                    <r type="polygon" id="end" pts="1,0,0.25|1,0,-0.25|1,1,-0.25|1,1,0.25"/>
                </r>
                <r id="1" pts="1,-0.5,0|1,-0.5,-1|1,1.5,-1|1,1.5,0">
                    <r type="polygon" id="front" pts="1.25,-0.5,1.53081e-17|1.25,-0.5,-1|1.25,1.5,-1|1.25,1.5,1.53081e-17"/>
                    <r type="polygon" id="back" pts="0.75,-0.5,-1|0.75,-0.5,0|0.75,1.5,0|0.75,1.5,-1"/>
                    <r type="polygon" id="top" pts="0.75,1.5,-1.53081e-17|1.25,1.5,1.53081e-17|1.25,1.5,-1|0.75,1.5,-1"/>
                    <r type="polygon" id="bottom" pts="0.75,-0.5,-1.53081e-17|0.75,-0.5,-1|1.25,-0.5,-1|1.25,-0.5,1.53081e-17"/>
                    <r type="polygon" id="start" pts="0.75,-0.5,-1.53081e-17|1.25,-0.5,1.53081e-17|1.25,1.5,1.53081e-17|0.75,1.5,-1.53081e-17"/>
                    <r type="polygon" id="end" pts="1.25,-0.5,-1|0.75,-0.5,-1|0.75,1.5,-1|1.25,1.5,-1"/>
                </r>
            </root>'''
        )

    def testThreePanelAngledJoint(self):
        """Test three panels meeting at varied angles (120° and 45°).
        
        Panel 0: 5 units along +X
        Panel 1: 5 units rotated 120° from panel 0's end
        Panel 2: 5 units rotated 45° attached at panel 0's start
        
        Creates a Y-shaped intersection with cap polygons at the joint.
        """
        self.assertXMLOutput(
            '''<root>
                <panel size="5" thickness="0.5"/>
                <panel size="5" rotate="120" thickness="0.5"/>
                <panel size="5" attach-id="0" rotate="45" thickness="0.5"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|5,0,0|5,1,0|0,1,0">
                    <r type="polygon" id="front" pts="0,0,0.25|5.10355,0,0.25|5.10355,1,0.25|0,1,0.25"/>
                    <r type="polygon" id="back" pts="4.56699,0,-0.25|0,0,-0.25|0,1,-0.25|4.56699,1,-0.25"/>
                    <r type="polygon" id="top" pts="0,1,-0.25|0,1,0.25|5.10355,1,0.25|4.56699,1,-0.25"/>
                    <r type="polygon" id="bottom" pts="0,0,-0.25|4.56699,0,-0.25|5.10355,0,0.25|0,0,0.25"/>
                    <r type="polygon" id="start" pts="0,0,-0.25|0,0,0.25|0,1,0.25|0,1,-0.25"/>
                </r>
                <r id="1" pts="5,0,0|2.5,0,-4.33013|2.5,1,-4.33013|5,1,0">
                    <r type="polygon" id="front" pts="5.0536,0,-0.407157|2.71651,0,-4.45513|2.71651,1,-4.45513|5.0536,1,-0.407157"/>
                    <r type="polygon" id="back" pts="2.28349,0,-4.20513|4.56699,0,-0.25|4.56699,1,-0.25|2.28349,1,-4.20513"/>
                    <r type="polygon" id="top" pts="4.56699,1,-0.25|5.0536,1,-0.407157|2.71651,1,-4.45513|2.28349,1,-4.20513"/>
                    <r type="polygon" id="bottom" pts="4.56699,0,-0.25|2.28349,0,-4.20513|2.71651,0,-4.45513|5.0536,0,-0.407157"/>
                    <r type="polygon" id="end" pts="2.71651,0,-4.45513|2.28349,0,-4.20513|2.28349,1,-4.20513|2.71651,1,-4.45513"/>
                    <r type="polygon" id="cap-top" pts="4.56699,1,-0.25|5.10355,1,0.25|5.0536,1,-0.407157"/>
                    <r type="polygon" id="cap-bottom" pts="5.0536,0,-0.407157|5.10355,0,0.25|4.56699,0,-0.25"/>
                </r>
                <r id="2" pts="5,0,0|8.53553,0,-3.53553|8.53553,1,-3.53553|5,1,0">
                    <r type="polygon" id="front" pts="5.10355,0,0.25|8.71231,0,-3.35876|8.71231,1,-3.35876|5.10355,1,0.25"/>
                    <r type="polygon" id="back" pts="8.35876,0,-3.71231|5.0536,0,-0.407157|5.0536,1,-0.407157|8.35876,1,-3.71231"/>
                    <r type="polygon" id="top" pts="5.0536,1,-0.407157|5.10355,1,0.25|8.71231,1,-3.35876|8.35876,1,-3.71231"/>
                    <r type="polygon" id="bottom" pts="5.0536,0,-0.407157|8.35876,0,-3.71231|8.71231,0,-3.35876|5.10355,0,0.25"/>
                    <r type="polygon" id="end" pts="8.71231,0,-3.35876|8.35876,0,-3.71231|8.35876,1,-3.71231|8.71231,1,-3.35876"/>
                </r>
            </root>'''
        )

    def testFivePanelComplexJoint(self):
        """Test five panels meeting at complex angles.
        
        Panel 0: 5 units along +X
        Panel 1: 5 units rotated 90° (perpendicular)
        Panel 2: 5 units rotated 135° attached at panel 0's start
        Panel 3: 5 units rotated 20° attached at panel 0's start
        Panel 4: 5 units rotated -40° attached at panel 0's start
        
        Creates a complex 5-way star intersection with pentagonal cap polygons.
        """
        self.assertXMLOutput(
            '''<root>
                <panel size="5" thickness="0.5"/>
                <panel size="5" rotate="90" thickness="0.5"/>
                <panel size="5" attach-id="0" rotate="135" thickness="0.5"/>
                <panel size="5" attach-id="0" rotate="20" thickness="0.5"/>
                <panel size="5" attach-id="0" rotate="-40" thickness="0.5"/>
            </root>''',
            '''<root>
                <r id="0" pts="0,0,0|5,0,0|5,1,0|0,1,0">
                    <r type="polygon" id="front" pts="0,0,0.25|4.90901,0,0.25|4.90901,1,0.25|0,1,0.25"/>
                    <r type="polygon" id="back" pts="4.39645,0,-0.25|0,0,-0.25|0,1,-0.25|4.39645,1,-0.25"/>
                    <r type="polygon" id="top" pts="0,1,-0.25|0,1,0.25|4.90901,1,0.25|4.39645,1,-0.25"/>
                    <r type="polygon" id="bottom" pts="0,0,-0.25|4.39645,0,-0.25|4.90901,0,0.25|0,0,0.25"/>
                    <r type="polygon" id="start" pts="0,0,-0.25|0,0,0.25|0,1,0.25|0,1,-0.25"/>
                </r>
                <r id="1" pts="5,0,0|5,0,-5|5,1,-5|5,1,0">
                    <r type="polygon" id="front" pts="5.25,0,-0.357037|5.25,0,-5|5.25,1,-5|5.25,1,-0.357037"/>
                    <r type="polygon" id="back" pts="4.75,0,-5|4.75,0,-0.603553|4.75,1,-0.603553|4.75,1,-5"/>
                    <r type="polygon" id="top" pts="4.75,1,-0.603553|5.25,1,-0.357037|5.25,1,-5|4.75,1,-5"/>
                    <r type="polygon" id="bottom" pts="4.75,0,-0.603553|4.75,0,-5|5.25,0,-5|5.25,0,-0.357037"/>
                    <r type="polygon" id="end" pts="5.25,0,-5|4.75,0,-5|4.75,1,-5|5.25,1,-5"/>
                </r>
                <r id="2" pts="5,0,0|1.46447,0,-3.53553|1.46447,1,-3.53553|5,1,0">
                    <r type="polygon" id="front" pts="4.75,0,-0.603553|1.64124,0,-3.71231|1.64124,1,-3.71231|4.75,1,-0.603553"/>
                    <r type="polygon" id="back" pts="1.28769,0,-3.35876|4.39645,0,-0.25|4.39645,1,-0.25|1.28769,1,-3.35876"/>
                    <r type="polygon" id="top" pts="4.39645,1,-0.25|4.75,1,-0.603553|1.64124,1,-3.71231|1.28769,1,-3.35876"/>
                    <r type="polygon" id="bottom" pts="4.39645,0,-0.25|1.28769,0,-3.35876|1.64124,0,-3.71231|4.75,0,-0.603553"/>
                    <r type="polygon" id="end" pts="1.64124,0,-3.71231|1.28769,0,-3.35876|1.28769,1,-3.35876|1.64124,1,-3.71231"/>
                    <r type="polygon" id="cap-top" pts="4.39645,1,-0.25|4.90901,1,0.25|5.4924,1,0.0868241|5.25,1,-0.357037|4.75,1,-0.603553"/>
                    <r type="polygon" id="cap-bottom" pts="4.75,0,-0.603553|5.25,0,-0.357037|5.4924,0,0.0868241|4.90901,0,0.25|4.39645,0,-0.25"/>
                </r>
                <r id="3" pts="5,0,0|9.69846,0,-1.7101|9.69846,1,-1.7101|5,1,0">
                    <r type="polygon" id="front" pts="5.4924,0,0.0868241|9.78397,0,-1.47518|9.78397,1,-1.47518|5.4924,1,0.0868241"/>
                    <r type="polygon" id="back" pts="9.61296,0,-1.94502|5.25,0,-0.357037|5.25,1,-0.357037|9.61296,1,-1.94502"/>
                    <r type="polygon" id="top" pts="5.25,1,-0.357037|5.4924,1,0.0868241|9.78397,1,-1.47518|9.61296,1,-1.94502"/>
                    <r type="polygon" id="bottom" pts="5.25,0,-0.357037|9.61296,0,-1.94502|9.78397,0,-1.47518|5.4924,0,0.0868241"/>
                    <r type="polygon" id="end" pts="9.78397,0,-1.47518|9.61296,0,-1.94502|9.61296,1,-1.94502|9.78397,1,-1.47518"/>
                </r>
                <r id="4" pts="5,0,0|8.83022,0,3.21394|8.83022,1,3.21394|5,1,0">
                    <r type="polygon" id="front" pts="4.90901,0,0.25|8.66953,0,3.40545|8.66953,1,3.40545|4.90901,1,0.25"/>
                    <r type="polygon" id="back" pts="8.99092,0,3.02243|5.4924,0,0.0868241|5.4924,1,0.0868241|8.99092,1,3.02243"/>
                    <r type="polygon" id="top" pts="5.4924,1,0.0868241|4.90901,1,0.25|8.66953,1,3.40545|8.99092,1,3.02243"/>
                    <r type="polygon" id="bottom" pts="5.4924,0,0.0868241|8.99092,0,3.02243|8.66953,0,3.40545|4.90901,0,0.25"/>
                    <r type="polygon" id="end" pts="8.66953,0,3.40545|8.99092,0,3.02243|8.99092,1,3.02243|8.66953,1,3.40545"/>
                </r>
            </root>'''
        )


if __name__ == '__main__':
    unittest.main()
