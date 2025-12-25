#!/usr/bin/env python
"""
Script to generate test expectation XML from panel XML input.

Reads panel XML from a file or stdin, processes it through the full layout pipeline,
and outputs the resulting structure as test expectation XML with corner points.

Usage:
    python generate_test_xml.py <input_file>
    python generate_test_xml.py input.xml
    python generate_test_xml.py -              # Read from stdin
    echo '<root>...</root>' | python generate_test_xml.py -
"""

import sys
import os
import textwrap

# Add the parent directory to the path so we can import the test fixtures
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from test_fixtures.xml_integration_test_tools import XMLIntegrationTestGenerator


def main():
    """Main entry point for the script."""
    if len(sys.argv) > 2:
        print("Usage: python generate_test_xml.py [input_file | -]")
        print("\nIf no input file is specified, uses 'test.xml' in the tests folder.")
        print("Use '-' to read XML from stdin.")
        print("\nExamples:")
        print("  python generate_test_xml.py input.xml")
        print("  python generate_test_xml.py")
        print("  python generate_test_xml.py -")
        print("  echo '<root>...</root>' | python generate_test_xml.py -")
        sys.exit(1)
    
    # Determine input source
    if len(sys.argv) == 2:
        if sys.argv[1] == '-':
            # Read from stdin
            xml_input = sys.stdin.read()
        else:
            # Read from file
            input_file = sys.argv[1]
            if not os.path.exists(input_file):
                print(f"Error: File '{input_file}' not found")
                sys.exit(1)
            try:
                with open(input_file, 'r') as f:
                    xml_input = f.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
    else:
        # Default to test.xml in the tests folder
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(tests_dir, "test.xml")
        print(f"No input file specified, using default: {input_file}\n")
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found")
            sys.exit(1)
        try:
            with open(input_file, 'r') as f:
                xml_input = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    # Generate test XML
    try:
        generator = XMLIntegrationTestGenerator(precision=6)
        xml_output = generator.generate_from_panel_xml(xml_input)
        
        # Clean up the input and output, preserving internal indentation
        xml_input_lines = xml_input.strip().split('\n')
        xml_output_lines = xml_output.strip().split('\n')
        
        # Format the output as a test function with proper indentation
        print("# Copy and paste this into your test class:\n")
        print("def test_generated(self):")
        print('    """Generated test from XML input."""')
        print('    xml_input = """' + xml_input_lines[0])
        for line in xml_input_lines[1:]:
            print('    ' + line)
        print('    """')
        print()
        print('    expected_xml = """' + xml_output_lines[0])
        for line in xml_output_lines[1:]:
            print('    ' + line)
        print('    """')
        print()
        print("    self.assertXMLOutput(xml_input, expected_xml)")
    except Exception as e:
        print(f"Error generating test XML: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
