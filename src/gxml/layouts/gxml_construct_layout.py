"""
    A layout that is used to layout each child element connected to the previous element, allowing for rotating them, and attaching
    elements to other elements to construct a larger shape of individual panels.
"""

import math
import copy

import mathutils.gxml_math as GXMLMath
from mathutils.quad_interpolator import QuadInterpolator
from gxml_types import *
from gxml_parsing_utils import GXMLParsingUtils
from layouts.gxml_base_layout import GXMLBaseLayout

class GXMLConstructLayout(GXMLBaseLayout):
    """ 
        Lays out elements in a construction style. Each panel starts attached to the end of the previous panel.
        You can change where the panel attaches to, rotate it around that axis, change the rotation axis, and adjust
        the size of each panel independently, or have those changes propagate to the next panel. Useful for generating
        things like building walls, fences, or floorplans of rooms.
    """
    
    layoutScheme = GXMLLayoutScheme.Construct
    
    def apply_default_layout_properties(self, element):
        super().apply_default_layout_properties(element)
        
        element.primaryAxis = Axis.X
        element.secondaryAxis = Axis.Y
        
        element.attachElement = None
        element.anchorElement = None
        element.attachedElements = []
        
        element.attachOffset = Offset(1.0, 0.0, 0.0)
        element.anchorOffset = Offset(0.0, 0.0, 0.0)
        
        element.attachStr = None
        element.anchorStr = None
        
        element.size0 = element.size1 = element.transform.scale
        element.offset0 = element.offset1 = element.transform.translation
        element.rotate = element.transform.rotation
        
    def parse_layout_attributes(self, element, ctx):
        super().parse_layout_attributes(element, ctx)
        
        attachId = ctx.getAttribute("attach-id", None)
        anchorId = ctx.getAttribute("anchor-id", None)
        
        if (attachId == "~" or attachId == None) and ctx.prevElem:
            attachId = ctx.prevElem.id
            
        element.attachElement = ctx.elementMap[attachId] if attachId in ctx.elementMap else None
        if element.attachElement:
            element.attachElement.attachedElements.append(element)
        
        element.anchorElement = ctx.elementMap[anchorId] if anchorId in ctx.elementMap else None
        
        attachTargetPointStr = element.attachElement.attachStr if element.attachElement else "right"
        attachPointStr = ctx.getAttribute("attach-to", None) or attachTargetPointStr or "right"
        anchorTargetPointStr = element.anchorElement.anchorStr if element.anchorElement else "left"
        anchorPointStr = ctx.getAttribute("anchor-to", None) or anchorTargetPointStr or "auto"
        
        primaryAxisStr = ctx.getAttribute("primary-axis", None)
        if primaryAxisStr:
            element.primaryAxis = Axis.parse(ctx.getAttribute("primary-axis", None))
        else:
            try: 
                element.primaryAxis = GXMLParsingUtils.infer_axis(attachPointStr, ctx)
                element.secondaryAxis = Axis.Y
            except:
                element.primaryAxis = Axis.X
                element.secondaryAxis = Axis.Y
                
        element.attachStr = ctx.getAttribute("attach-point", "right")
        element.attachOffset = GXMLParsingUtils.parse_offset_vector(attachPointStr, ctx, element.primaryAxis, element.secondaryAxis)
        element.anchorStr = ctx.getAttribute("anchor-point", "auto")
        element.anchorOffset = GXMLParsingUtils.parse_offset_vector(anchorPointStr, ctx, element.primaryAxis, element.secondaryAxis)
        
        element.transform.pivot = GXMLParsingUtils.parse_offset_vector(ctx.getAttribute("pivot", "left"), ctx, element.primaryAxis, element.secondaryAxis)
            
        rotate = GXMLParsingUtils.parse_transformation_vector(ctx.getAttribute("rotate", "0"), ctx, element.primaryAxis, 0, 0)
        size = GXMLParsingUtils.parse_transformation_vector(ctx.getAttribute("size", "1"), ctx, element.primaryAxis, 1, 1)
        offset = GXMLParsingUtils.parse_transformation_vector(ctx.getAttribute("offset", "0"), ctx, element.primaryAxis, 0, 1)
        
        element.size0, element.size1 = GXMLParsingUtils.parse_individual_transformation_attributes(ctx, size, size, "width", "height", "depth")
        element.offset0, element.offset1 = GXMLParsingUtils.parse_individual_transformation_attributes(ctx, offset, offset, "x", "y", "z")
        element.rotate, _ = GXMLParsingUtils.parse_individual_transformation_attributes(ctx, rotate, rotate, "pitch", "yaw", "roll")
        
    def find_intersection_point(self, transform, direction, point):
        p1 = transform.transform_point((0,0,0))
        p2 = transform.transform_point((1,0,0))
        return GXMLMath.find_intersection_ray_to_segment(point, direction, p1, p2)
    
    def find_auto_anchor_point(self, element, attachTransform, anchorTransform, attachPoint, anchorOffset):
        p1 = attachTransform.transform_point((0,0,0))
        p2 = attachTransform.transform_point((1,0,0))
        p3 = attachTransform.transform_point((0,1,0))
        
        anchorP1 = anchorTransform.transform_point((0,0,0))
        anchorP2 = anchorTransform.transform_point((1,0,0))
        
        angle = element.transform.rotation[1]
        d1 = GXMLMath.safe_normalize(p2 - p1)
        d2 = GXMLMath.safe_normalize(p3 - p1)
        d3 = GXMLMath.rotate_vector(d1, d2, angle)
        
        intersectionPoint = self.find_intersection_point(anchorTransform, d3, attachPoint)
        
        if intersectionPoint is None:
            angle1 = GXMLMath.angle_between(d1, GXMLMath.safe_normalize(anchorP1 - attachPoint))
            angle2 = GXMLMath.angle_between(d1, GXMLMath.safe_normalize(anchorP2 - attachPoint))
            
            angleDiff1 = abs(angle1 - angle)
            angleDiff2 = abs(angle2 - angle)
            
            if(angleDiff1 < angleDiff2):
                return anchorP1
            else:
                return anchorP2
        else:
            return intersectionPoint
        
    def calculate_anchor_matrix(self, element, attachTransform, attachOffset, anchorTransform, anchorOffset, height):
        attachPoint = attachTransform.transform_point(attachOffset)
        
        if anchorOffset.auto:
            anchorPoint = self.find_auto_anchor_point(element, attachTransform, anchorTransform, attachPoint, anchorOffset)
        else:
            anchorPoint = anchorTransform.transform_point(anchorOffset)
        
        right = attachTransform.right() if attachTransform else element.transform.right()
        
        offset = anchorPoint - attachPoint
        distance = GXMLMath.length(offset)
        
        if(distance == 0):
            # With unit square architecture, still need to maintain thickness scale
            thickness = element.transform.scale[2] if element.transform.scale is not None else 1.0
            return GXMLMath.scale_matrix(0, 1, thickness)
        else:
            direction = GXMLMath.normalize(offset)
            determinant = GXMLMath.cross(right, direction)
            
            angle = math.degrees(math.atan2(determinant[1], GXMLMath.dot_product(right, direction)))
            
            # With unit square architecture: quad is [(0,0,0), (1,0,0), (1,1,0), (0,1,0)] in local space
            # The anchor matrix needs to scale by (distance, height, thickness) where thickness
            # is preserved from the original transform scale
            thickness = element.transform.scale[2] if element.transform.scale is not None else 1.0
            
            anchorRotationMatrix = GXMLMath.rot_matrix(0, angle, 0)
            anchorScaleMatrix = GXMLMath.scale_matrix(distance, height, thickness)
            return GXMLMath.mat_mul(anchorScaleMatrix, anchorRotationMatrix)
        
    def pre_layout_element(self, element):
        super().pre_layout_element(element)
        
        size0 = self.expand_size(element.size0, element.attachElement, element.attachOffset)
        size1 = self.expand_size(element.size1, element.anchorElement, element.anchorOffset, size0[1])
        self.build_local_transform(element, size0, size1)
        self.build_world_transform(element, max(size0[1], size1[1]))
        
    def layout_element(self, element):
        super().layout_element(element)
        
        def is_size_dependent(size):
            return any(isinstance(v, str) and "*" in v for v in size)
        
        # We should have re-calculated all the independent sizes in pre-layout. We want to now recalculate the dependent sizes
        # based on the elements they depend on which were previously calculated in pre-layout.
        if is_size_dependent(element.size0) or is_size_dependent(element.size1):
            attachElement0, attachOffset0, attachElement1, attachOffset1 = self.find_attached_elements(element)
            size0 = self.expand_size(element.size0, attachElement0, attachOffset0)
            size1 = self.expand_size(element.size1, attachElement1, attachOffset1, size0[1])
        
            self.build_local_transform(element, size0, size1)
            self.build_world_transform(element, max(size0[1], size1[1]))
    
    def find_attached_elements(self, element):
        upstreamAttachElement = element.attachElement
        downstreamAnchorElement = element.anchorElement
        
        for child in element.parent.children:
            # To find the upstream attachElement without actually using the attachElement of this element, we need
            # to actually check the anchorElement. As anchors attach to attach points of elements, and vice versa
            if upstreamAttachElement == None and child.anchorElement == element:
                upstreamAttachElement = child
            if downstreamAnchorElement == None and child.attachElement == element:
                downstreamAnchorElement = child
        
        return upstreamAttachElement, element.attachOffset, downstreamAnchorElement, element.anchorOffset
        
    def expand_size(self, size, targetElem, targetElemOffset, defaultSize = 1.0):
        newSize = [size[0],size[1],size[2]]
        
        for i, v in enumerate(size):
            if isinstance(v, str) and "*" in v:
                if targetElem:
                    p0 = targetElem.transform_point((targetElemOffset[0], targetElemOffset[1], 0))
                    p1 = targetElem.transform_point((targetElemOffset[0], 1, 0))
                    newSize[i] = p1[1] - p0[1]
                else:
                    newSize[i] = defaultSize
                
        return np.array(newSize)
    
    def build_local_transform(self, element, size0, size1):
        s0, s1, sMax = size0[1], size1[1], max(size0[1], size1[1])
        sizeRatio = np.array([s0, s1]) / sMax if sMax > 0 else np.array([0, 0])
        size = (max(size0[0], size1[0]), max(size0[1], size1[1]), max(size0[2], size1[2]))
        
        offsetRatioX = np.array([element.offset0[0], element.offset1[0]]) / size[0] if size[0] > 0 else np.array([0, 0])
        offsetRatioY = np.array([element.offset0[1], element.offset1[1]]) / size[1] if size[1] > 0 else np.array([0, 0])
        offsetRatioZ = np.array([element.offset0[2], element.offset1[2]]) / size[2] if size[2] > 0 else np.array([0, 0])
        
        element.quad_interpolator = QuadInterpolator((offsetRatioX[0], offsetRatioY[0], offsetRatioZ[0]), 
                                (offsetRatioX[0] + 1, offsetRatioY[1], offsetRatioZ[0]),
                                (offsetRatioX[1] + 1,offsetRatioY[1]+sizeRatio[1], offsetRatioZ[1]), 
                                (offsetRatioX[1], offsetRatioY[0]+sizeRatio[0], offsetRatioZ[1]))
        element.transform.apply_local_transformations([0,0,0], element.rotate, size)
    
    def build_world_transform(self, element, height):
        attachTransform = copy.copy(element.attachElement.transform) if element.attachElement else None
        anchorTransform = copy.copy(element.anchorElement.transform) if element.anchorElement else None
        
        if anchorTransform:
            element.transform.localTransformationMatrix = self.calculate_anchor_matrix(element, attachTransform, element.attachOffset, anchorTransform, element.anchorOffset, height)
        
        if attachTransform:
            attachMatrix = GXMLMath.translate_matrix(element.attachOffset)
            attachTransform.transformationMatrix = GXMLMath.mat_mul(attachMatrix, attachTransform.transformationMatrix)
            attachTransform.localTransformationMatrix = GXMLMath.mat_mul(attachMatrix, attachTransform.localTransformationMatrix)
            
        element.transform.recalculate(attachTransform)