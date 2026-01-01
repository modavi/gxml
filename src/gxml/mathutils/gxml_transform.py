"""
GXMLTransform - Transformation matrix handling for GXML elements.
"""

import numpy as np
import mathutils.gxml_math as GXMLMath


class GXMLTransform:
    def __init__(self):
        self.pivot = (0.0, 0.0, 0.0)
        self.inheritScale = False
        
        self.localTransformationMatrix = GXMLMath.identity()
        self.transformationMatrix = GXMLMath.identity()
        
        # We also store these components individually in the case of needing to know the original
        # values for translation and rotation when a 0 scale has been applied in any axis
        self.translation = (0.0, 0.0, 0.0)
        self.rotation = (0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)
        
    def apply_local_transformations(self, translate, rotate, scale):
        self.localTransformationMatrix = GXMLMath.build_transform_matrix(translate, rotate, scale)
        self.translation = translate
        self.rotation = rotate
        self.scale = scale
        
    def decompose(self):
        return self.translation, self.rotation, self.scale
        
    def right(self):
        return self.transform_direction((1.0, 0.0, 0.0))

    def left(self):
        return self.transform_direction((-1.0, 0.0, 0.0))

    def up(self):
        return self.transform_direction((0.0, 1.0, 0.0))

    def down(self):
        return self.transform_direction((0.0, -1.0, 0.0))

    def forward(self):
        return self.transform_direction((0.0, 0.0, 1.0))

    def backward(self):
        return self.transform_direction((0.0, 0.0, -1.0))
    
    def sceneToLocal(self, point):
        if np.isclose(GXMLMath.determinant(self.transformationMatrix), 0):
            return [0,0,0]
        
        inv = GXMLMath.invert(self.transformationMatrix)
        local_point = GXMLMath.transform_point(point, inv)
        # Already in 0-1 space, just return as list
        return [local_point[0], local_point[1], local_point[2]]

    def transform_point(self, point):
        return GXMLMath.transform_point(point, self.transformationMatrix)
    
    def transform_direction(self, vec):
        return GXMLMath.transform_direction(vec, self.transformationMatrix)
    
    def recalculate(self, parentTransform):
        localMatrix = GXMLMath.mat_mul(GXMLMath.translate_matrix(tuple(-x for x in self.pivot)), self.localTransformationMatrix)
                
        if(parentTransform):
            scaleInheritance = GXMLMath.ScaleInheritance.Default if self.inheritScale else GXMLMath.ScaleInheritance.Ignore
            
            parentWorldTransform = parentTransform.transformationMatrix
            parentLocalTransform = parentTransform.localTransformationMatrix
            
            if np.isclose(parentTransform.scale, 0.0).any():
                parentWorldTransform = GXMLMath.build_transform_matrix(parentTransform.translation, parentTransform.rotation, (1,1,1))
                parentLocalTransform = GXMLMath.build_transform_matrix(parentTransform.translation, parentTransform.rotation, (1,1,1))
            
            self.transformationMatrix = GXMLMath.combine_transform(localMatrix, parentWorldTransform, parentLocalTransform, scaleInheritance)
        else:
            self.transformationMatrix = localMatrix
