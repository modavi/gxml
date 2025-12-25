from render_engines.base_render_context import BaseRenderContext
import hou

class HoudiniRenderContext(BaseRenderContext):
    def __init__(self, node):
        self.node = node
        self.geo = node.geometry()
        self.geoMap = {}
        
    def create_poly(self, id, points, geoKey = None):
        geo = self.get_or_create_geo(geoKey)
        
        poly = geo.createPolygon()
        for pos in points:
            point = geo.createPoint()
            point.setPosition(pos)
            poly.addVertex(point)
        
        attrib = geo.findPrimAttrib("id")
        if attrib is None:
            attrib = geo.addAttrib(hou.attribType.Prim, "id", "")
            
        poly.setAttribValue(attrib, id)
        
    def create_line(self, id, points):
        poly = self.geo.createPolygon(False)
        for pos in points:
            point = self.geo.createPoint()
            point.setPosition(pos)
            poly.addVertex(point)
    
    def get_geo(self, key):
        if key == None:
            return self.geo
        
        if key in self.geoMap:
            return self.geoMap[key]
            
        return None
            
    def get_or_create_geo(self, key):
        if key == None:
            return self.geo
        
        if key not in self.geoMap:
            self.geoMap[key] = hou.Geometry()
            
        return self.geoMap[key]
    
    def combine_all_geo(self):
        for key, geo in self.geoMap.items():
            self.geo.merge(geo)
            
    def unify_geo(self, geoKey):
        geo = self.get_geo(geoKey)
        
        if not geo:
            return
        
        verb = hou.sopNodeTypeCategory().nodeVerb("boolean::2.0")
        verb.setParms({
            "booleanop": 0,
            "asurface": 0,
            "bsurface": 0,
        })
        verb.execute(geo, [geo])
        
        # Get border edges
        edgeGroup = ""
        for edge in geo.globEdges("*"):
            if len(edge.prims()) == 1:
                edgeGroup += " " + edge.edgeId()
        
        verb = hou.sopNodeTypeCategory().nodeVerb("dissolve::2.0")
        verb.setParms({
            "group": edgeGroup,
            "invertsel": 1,
            "reminlinepts": 0,
            "deldegeneratebridges": 1,
            "boundarycurves": 1
        })
        verb.execute(geo, [geo])