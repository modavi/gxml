"""
    Provides types and utilities for GXML layout and transformation.
"""

import mathutils.gxml_math as GXMLMath
from mathutils.gxml_transform import GXMLTransform
import numpy as np

class GXMLLayoutScheme:
    Construct = "construct"
    Stack = "stack"
    Fixed = "fixed"

class Layout:
    Horizontal = 1
    Vertical = 2
    
    def parse(str):
        if str == "horizontal":
            return Layout.Horizontal
        if str == "vertical":
            return Layout.Vertical
        
class Offset:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.auto = False

    @staticmethod
    def auto():
        offset = Offset(0, 0, 0)
        offset.auto = True
        return offset

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("Index out of range")

    def __len__(self):
        return 3
        
class Axis:
    X = 1
    Y = 2
    Z = 3
    Undefined = 4
    
    def parse(axisStr):
        if axisStr.lower() == "x":
            return Axis.X
        if axisStr.lower() == "y":
            return Axis.Y
        if axisStr.lower() == "z":
            return Axis.Z
        
        return Axis.Undefined
        
class Side:
    Undefined = 0
    Bottom = 1
    Top = 2
    Left = 3
    Right = 4
    Front = 5
    Back = 6
    
    Center = 7
    
    def offset(side):
        x = y = z = 0
        
        if side == Side.Left:
            x = 0
        if side == Side.Right:
            x = 1
        
        if side == Side.Bottom:
            y = 0
        if side == Side.Top:
            y = 1
            
        if side == Side.Front:
            z = 0
        elif side == Side.Back:
            z = 1
            
        if side == Side.Center:
            x = y = z = 0.5
        
        return Offset(x,y,z)
    
    def dir(side):
        if side == Side.Bottom:
            return 0
        if side == Side.Top:
            return 1
        
        if side == Side.Left:
            return 0
        if side == Side.Right:
            return 1
        
        if side == Side.Front:
            return 0
        if side == Side.Back:
            return 1
        
        if side == Side.Center:
            return 0.5
        
        return 0
    
    def parse(sideStr):
        if sideStr == "bottom":
            return Side.Bottom
        if sideStr == "top":
            return Side.Top
        if sideStr == "left":
            return Side.Left
        if sideStr == "right":
            return Side.Right
        if sideStr == "front":
            return Side.Front
        if sideStr == "back":
            return Side.Back
        if sideStr == "center":
            return Side.Center
        
        return Side.Undefined

class Size:
    def __init__(self, width, height, stretch = False):
        self.width = width
        self.height = height
        self.stretch = stretch