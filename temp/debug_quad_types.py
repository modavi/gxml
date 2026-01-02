import cProfile
import pstats
import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src/gxml')
sys.path.insert(0, '/Users/morgan/Projects/gxml-web/src')

from gxml_engine import GXMLEngine
from gxml_web.json_render_engine import JSONRenderEngine

def test_quad_types():
    """Debug what types are being passed to create_transform_matrix_from_quad"""
    from gxml.mathutils.gxml_math import create_transform_matrix_from_quad
    
    # Check what gets passed in
    original_func = create_transform_matrix_from_quad
    def wrapped(points):
        print(f"Points type: {type(points)}")
        print(f"p0 type: {type(points[0])}, value: {points[0]}")
        print(f"p0[0] type: {type(points[0][0])}")
        return original_func(points)
    
    import gxml.mathutils.gxml_math as gxml_math
    gxml_math.create_transform_matrix_from_quad = wrapped
    
    xml = '''
    <Root>
        <Panel name="test" width="100" height="100"/>
    </Root>
    '''
    engine = GXMLEngine()
    engine.load_string(xml)
    engine.layout()
    engine.render(JSONRenderEngine())

test_quad_types()
