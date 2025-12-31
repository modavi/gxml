"""
Provides helper methods for parsing GXML elements and attributes.
This includes parsing offset vectors, transformation vectors, and individual transformation attributes.
Also includes methods for inferring axes from strings and parsing offset vectors with support for auto values.
"""

from gxml_types import *

class GXMLParsingUtils:
    """Provides helper methods for parsing GXML elements and attributes."""
    
    def _parse_corner(cornerStr):
        """Parse corner names like 'top-left', 'bottom-right' into Offset.
        Returns None if not a valid corner name."""
        corners = {
            "top-left": Offset(0, 1, 0),
            "top-right": Offset(1, 1, 0),
            "bottom-left": Offset(0, 0, 0),
            "bottom-right": Offset(1, 0, 0),
            # Also support reversed order
            "left-top": Offset(0, 1, 0),
            "right-top": Offset(1, 1, 0),
            "left-bottom": Offset(0, 0, 0),
            "right-bottom": Offset(1, 0, 0),
        }
        return corners.get(cornerStr.lower())
    
    def parse_offset_vector(offsetStr, ctx, primaryAxis = Axis.X, secondaryAxis = Axis.Y):
        if offsetStr == "auto":
            return Offset.auto()
        
        # Check for corner names like "top-left", "bottom-right"
        corner = GXMLParsingUtils._parse_corner(offsetStr)
        if corner is not None:
            return corner
        
        tokens = offsetStr.split(",")
        
        supportedStrings = [("left","right","center"), ("top","bottom","center"), ("front","back","center")]
        
        for i in range(0, len(tokens)):
            parsedSide = Side.parse(tokens[i])
            
            if parsedSide == Side.Undefined:
                tokens[i] = ctx.eval(tokens[i])
            else:
                if len(tokens) > 1 and tokens[i] not in supportedStrings[i]:
                    raise ValueError(f"Invalid side string: {tokens[i]} for offset vector index {i}")
                
                tokens[i] = Side.dir(parsedSide)    
                
        if len(tokens) == 1:
            if parsedSide != Side.Undefined:
                return Side.offset(Side.parse(offsetStr))
            else:
                return Offset(tokens[0], 0, 0) if primaryAxis == Axis.X else Offset(0, tokens[0], 0) if primaryAxis == Axis.Y else Offset(0, 0, tokens[0])
        elif len(tokens) == 2:
            returnVal = [0,0,0]
            
            if primaryAxis == Axis.X:
                returnVal[0] = tokens[0]
            if primaryAxis == Axis.Y:
                returnVal[1] = tokens[0]
            if primaryAxis == Axis.Z:
                returnVal[2] = tokens[0]
                
            if secondaryAxis == Axis.X:
                returnVal[0] = tokens[1]
            if secondaryAxis == Axis.Y:
                returnVal[1] = tokens[1]
            if secondaryAxis == Axis.Z:
                returnVal[2] = tokens[1]
            
            return Offset(returnVal[0], returnVal[1], returnVal[2])
        elif len(tokens) == 3:
            return Offset(tokens[0], tokens[1], tokens[2])
        
        return Offset(0,0,0)
    
    def parse_transformation_vector(str, ctx, axis, defaultVal, isScale):
        tokens = str.split(",")
        
        output = (defaultVal, defaultVal, defaultVal)
        if len(tokens) == 3:
            output = (ctx.eval(tokens[0]), ctx.eval(tokens[1]), ctx.eval(tokens[2]))
        if len(tokens) == 2:
            output = (ctx.eval(tokens[0]), ctx.eval(tokens[1]), defaultVal)
        if len(tokens) == 1:
            val = ctx.eval(tokens[0])
            if axis == Axis.X:
                output = (val, defaultVal, defaultVal) if isScale else (defaultVal, val, defaultVal)
            if axis == Axis.Y:
                output = (defaultVal, val, defaultVal) if isScale else (val, defaultVal, defaultVal)
            if axis == Axis.Z:
                output = (defaultVal, defaultVal, val) if isScale else (defaultVal, defaultVal, val)
        
        return output
    
    def parse_individual_transformation_attributes(ctx, transformVector0, transformVector1, xArg, yArg, zArg):
        def process_token(token, ctx):
            if "*" in token:
                return token
            return ctx.eval(token)
        
        def parse_val(arg, defaultVal0, defaultVal1, ctx):
            v1, v2 = defaultVal0, defaultVal1
            
            strVal = ctx.getAttribute(arg, None)
            if strVal:
                tokens = strVal.split(":")
                if len(tokens) == 1:
                    tokens.append(tokens[0])
                v1, v2 = process_token(tokens[0], ctx), process_token(tokens[1], ctx)
            
            return v1, v2
        
        xVal0, xVal1 = parse_val(xArg, transformVector0[0], transformVector1[0], ctx)
        yVal0, yVal1 = parse_val(yArg, transformVector0[1], transformVector1[1], ctx)
        zVal0, zVal1 = parse_val(zArg, transformVector0[2], transformVector1[2], ctx)
            
        return (xVal0, yVal0, zVal0), (xVal1, yVal1, zVal1)

    def infer_axis(str, ctx):
        tokens = str.split(",")
        
        for i in range(0, len(tokens)):
            tokens[i] = Side.parse(tokens[i])
        
        for token in tokens:
            if token == Side.Center:
                continue
            
            if token == Side.Top or token == Side.Bottom:
                return Axis.Y
            if token == Side.Left or token == Side.Right:
                return Axis.X
            if token == Side.Front or token == Side.Back:
                return Axis.Z
        
        raise ValueError(f"Invalid axis inference string: {str}")