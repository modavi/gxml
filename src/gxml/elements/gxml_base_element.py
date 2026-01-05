"""
    A layout element is a base class for all the elements that can be laid out in a gxml structure. It supports up to 3
    dimensions and contains base level functionality for how an element should lay itself out in the overall structure.
"""

from abc import abstractmethod
import copy
from enum import Enum, auto
from collections import deque

from gxml_types import *
from profiling import *
# Import GXMLTransform directly to avoid circular import issues with import *
from mathutils.gxml_transform import GXMLTransform

class TraversalDirection(Enum):
    """ Direction for element traversal """
    TopDownDepthFirst = auto()    # Process parent before children, depth-first traversal
    TopDownBreadthFirst = auto()  # Process parent before children, breadth-first traversal
    BottomUp = auto()             # Process children before parent (depth-first)
    
class GXMLLayoutElement(object):
    """
        A base class for all elements that can be laid out in a GXML structure.
        This class provides the basic properties and methods needed for layout,
        including handling child elements, visibility, transformations, and layout schemes.
    """
    
    def __init__(self):
        self.id = None
        # Used for elements that generate additional dynamic elements, such as blocks which will create panels for each side.
        # The dynamically generated elements will all share the same ID as the generator, but will get unique subIds.
        self.subId = None
        
        self.parent = None
        self.children = []
        self.dynamicChildren = []
        
        self.isVisible = True
        self.isVisibleSelf = True
        
        self.transform = GXMLTransform()
        
        self.childLayoutScheme = None
        
    def add_child(self, child):
        assert child != self
        assert child.parent == None
        
        child.parent = self
        self.children.append(child)
        
    def add_dynamic_child(self, dynamicElement):
        assert dynamicElement != self
        assert dynamicElement.parent == None
        
        dynamicElement.parent = self
        self.dynamicChildren.append(dynamicElement)
        
    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
        if child in self.dynamicChildren:
            self.dynamicChildren.remove(child)
            
    def siblings(self):
        if self.parent == None:
            return []
        
        return self.parent.children
    
    def root(self):
        if self.parent == None:
            return self
        return self.parent.root()
            
    def iterate(self, direction=TraversalDirection.TopDownBreadthFirst, includeDynamicChildren = False):
        """
            Traverses all elements in the hierarchy, calling the provided callback for each element.
            
            Args:
                direction (TraversalDirection): Direction of traversal:
                                            - TopDownDepthFirst: Parent first, then children (depth-first)
                                            - TopDownBreadthFirst: Level by level traversal
                                            - BottomUp: Children first, then parent
        """
        
        if direction == TraversalDirection.BottomUp:
            for child in self.children:
                yield from child.iterate(direction)
                
            if includeDynamicChildren:
                for child in self.dynamicChildren:
                    yield from child.iterate(direction)
                    
            yield self
        
        elif direction == TraversalDirection.TopDownDepthFirst:
            yield self
            
            for child in self.children:
                yield from child.iterate(direction)
                
            if includeDynamicChildren:
                for child in self.dynamicChildren:
                    yield from child.iterate(direction)
        
        elif direction == TraversalDirection.TopDownBreadthFirst:
            queue = deque([self])
            
            while queue:
                current = queue.popleft()
                
                yield current
                
                queue.extend(current.children)
                
                if includeDynamicChildren:
                    queue.extend(current.dynamicChildren)
    
    def parse(self, ctx):
        self.id = ctx.desc.get("id") or str(ctx.childIndex)
        self.id = self.id.replace("#", str(ctx.childIndex))
        
        self.childLayoutScheme = ctx.getAttribute("layout", self.childLayoutScheme)
        self.isVisible = ctx.getAttribute("visible", "true").lower() == "true"
        self.isVisibleSelf = ctx.getAttribute("visible-self", "true").lower() == "true"
        
    @abstractmethod
    def render(self, renderContext):
        """ Render this element out as actual points and primitives in the world"""
        
    @abstractmethod
    def on_pre_layout(self):
        """ Called before the layout pass to allow the element to prepare itself for layout """
        pass
    
    @abstractmethod
    def on_layout(self):
        """ Called after the layout pass to allow the element to finalize its layout """
        pass
    
    @abstractmethod
    def on_post_layout(self):
        """ Called after the post-layout pass to allow the element to finalize its layout """
        pass
        
    def transform_point(self, point):
        return self.transform.transform_point(point)
    
    def recalculate_transform(self):
        """Recalculate world transform based on parent.
        
        Call this after modifying transform.apply_local_transformations() to update
        the world-space transformation matrix.
        """
        self.transform.recalculate(self.parent.transform if self.parent else None)
    
    def get_transform(self):
        """Get the current transformation components.
        
        Returns:
            Tuple of (translation, rotation, scale) where each is a tuple of 3 floats
        """
        return self.transform.decompose()
        
    def clone(self):
        cloneCopy = copy.copy(self)
        cloneCopy.transform = copy.copy(cloneCopy.transform)
        
        # Make sure we clear out the parent of this clone, as even though it's a copy, it wont be
        # in any lists therefore its parent should be null until we add it as a child to something.
        cloneCopy.parent = None
        cloneCopy.children = []
        cloneCopy.dynamicChildren = []
        return cloneCopy