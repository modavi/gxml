from abc import abstractmethod

class BaseRenderContext(object):
    def render_hierarchy(self, element):
        self.conditional_render(element)
    
    def conditional_render(self, element):
        if element.isVisible:
            if element.isVisibleSelf:
                self.pre_render(element)
                element.render(self)
                self.post_render(element)
                
            for child in element.children:
                self.conditional_render(child)
    
    def pre_render(self, element):
        pass    
        
    def post_render(self, element):
        pass
    
    def combine_all_geo(self):
        pass
    
    def get_or_create_geo(self, key):
        pass
    
    def unify_geo(self, geoKey):
        pass
        
    @abstractmethod
    def create_poly(self, id, points, geoKey = None):
        """Creates a polygon with the given id and points."""
        
    @abstractmethod
    def create_line(self, id, points, geoKey = None):
        """Creates a line with the given id and points."""    