"""
GXML Engine - Main entry point for processing GXML content.

This module provides the primary API for parsing, laying out, and rendering
GXML (Geometric XML) content.

Mental Model (analogous to HTML/DOM):
    
    HTML                    GXML
    ----                    ----
    HTML string      →      GXML string (markup)
    DOM              →      GOM (Geometry Object Model)
    Rendered page    →      Rendered mesh (binary for WebGL)
    
    The pipeline mirrors web browsers:
    1. Parse: GXML string → GOM (element tree with panels, groups, etc.)
    2. Layout: Compute positions, intersections, face segmentation, geometry
    3. Render: GOM → mesh output (packed binary data for GPU)
    
    Just as you can manipulate the DOM after parsing HTML, you can inspect
    and traverse the GOM after parsing GXML. The GOM contains the full
    geometric state: panel transforms, intersection points, face segments,
    and computed polygon geometry.

Usage:
    from gxml import run, GXMLConfig
    
    # Simple usage - returns both GOM and rendered mesh
    result = run(xml_string)
    gom = result.gom    # The Geometry Object Model (element tree)
    mesh = result.mesh  # Rendered mesh (bytes for WebGL)
    
    # With configuration
    config = GXMLConfig(backend='c')
    result = run(xml_string, config=config)
    
    # Custom mesh render context
    from render_engines.binary_render_context import BinaryRenderContext
    ctx = BinaryRenderContext(shared_vertices=True, include_endpoints=True)
    result = run(xml_string, mesh_render_context=ctx)
    
    # Skip mesh rendering - just get the GOM
    result = run(xml_string, mesh_render_context=False)
    gom = result.gom  # Full GOM with all computed geometry
    
    # With profiling
    result = run(xml_string, profile=True)
    print(result.timings)
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Literal, Dict, Union
from pathlib import Path
import time

from lxml import etree

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from profiling import (
    reset_profile, 
    get_profile_results,
    perf_marker,
)


# =============================================================================
# XSD Schema Validation
# =============================================================================

# Load schema once at module import
_SCHEMA_PATH = Path(__file__).parent.parent.parent / 'misc' / 'gxml.xsd'
_SCHEMA: Optional[etree.XMLSchema] = None


def _get_schema() -> etree.XMLSchema:
    """Lazy-load the XSD schema."""
    global _SCHEMA
    if _SCHEMA is None:
        schema_doc = etree.parse(str(_SCHEMA_PATH))
        _SCHEMA = etree.XMLSchema(schema_doc)
    return _SCHEMA


def validate_xml(xml: str) -> None:
    """
    Validate XML against the GXML schema.
    
    Raises:
        etree.XMLSyntaxError: If XML is malformed
        etree.DocumentInvalid: If XML doesn't match schema
    """
    schema = _get_schema()
    doc = etree.fromstring(xml.encode())
    schema.assertValid(doc)


# =============================================================================
# Configuration
# =============================================================================

Backend = Literal['cpu', 'c', 'taichi', 'gpu']


@dataclass
class GXMLConfig:
    """
    Configuration options for the GXML processing pipeline.
    
    Attributes:
        backend: Solver backend for intersection/face/geometry computation.
            - 'cpu': Pure Python (default, always available)
            - 'c': C extension (fast, requires compilation)
            - 'taichi': GPU-accelerated (requires Taichi)
            - 'gpu': Legacy GPU mode
        
        mesh_render_context: Render context for mesh output. Can be:
            - None (default): Use default BinaryRenderContext
            - A render context instance (e.g., BinaryRenderContext with options)
            - False: Skip mesh rendering entirely
        
        profile: Enable timing profiling of pipeline stages.
            When True, returns timing data from all instrumented code sections.
            Markers are placed throughout the codebase using perf_marker context manager.
    """
    backend: Backend = 'cpu'
    mesh_render_context: Any = None
    profile: bool = False


@dataclass
class GXMLResult:
    """
    Result from running the GXML pipeline.
    
    Attributes:
        gom: The Geometry Object Model - the parsed and laid-out element tree.
            Analogous to the DOM in web browsers. Contains panels, groups,
            and all computed geometric state (transforms, intersections,
            face segments, polygon geometry).
        mesh: The rendered mesh (bytes from BinaryRenderContext).
            Analogous to the rendered page in a browser. Ready for GPU upload.
            None if mesh_render_context=False.
        timings: Timing data from profiled code sections (if profile=True).
            Each key is a marker name, value contains count, total_ms, avg_ms, min_ms, max_ms.
        stats: Statistics about the processed content (panel_count, intersection_count, etc.).
    """
    gom: Any  # Geometry Object Model (element tree)
    mesh: Any = None  # Rendered mesh (bytes)
    timings: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, int]] = None
    
    # Backwards compatibility aliases
    @property
    def root(self) -> Any:
        """Alias for gom (backwards compatibility)."""
        return self.gom
    
    @property
    def output(self) -> Any:
        """Alias for mesh (backwards compatibility)."""
        return self.mesh


# =============================================================================
# Main API
# =============================================================================

# Sentinel for "argument not provided" in run() kwargs
_NOT_PROVIDED = object()


def run(
    xml: str,
    config: Optional[GXMLConfig] = None,
    *,
    # Convenience kwargs that override config
    backend: Optional[Backend] = None,
    mesh_render_context: Any = _NOT_PROVIDED,
    profile: Optional[bool] = None,
) -> GXMLResult:
    """
    Run the GXML processing pipeline.
    
    This is the main entry point for processing GXML content. It handles:
    1. Parse: GXML string → GOM (Geometry Object Model)
    2. Layout: Compute positions, intersections, geometry on the GOM
    3. Render: GOM → mesh output (binary data for GPU)
    
    Args:
        xml: The GXML string to process.
        config: Configuration options (GXMLConfig instance).
        backend: Override config.backend.
        mesh_render_context: Override config.mesh_render_context.
            - None (default): Use default BinaryRenderContext
            - A render context instance: Use that context
            - False: Skip mesh rendering (just build GOM)
        profile: Override config.profile.
    
    Returns:
        GXMLResult containing:
        - gom: The Geometry Object Model (element tree with computed geometry)
        - mesh: Rendered mesh bytes (or None if mesh_render_context=False)
        - timings: Profile data (if profile=True)
        - stats: Panel/intersection counts
    
    Examples:
        # Basic usage - get both GOM and mesh
        result = run('<root><panel/></root>')
        gom = result.gom    # Element tree
        mesh = result.mesh  # Binary mesh data
        
        # With backend selection
        result = run(xml, backend='c')
        
        # Custom render context
        from render_engines.binary_render_context import BinaryRenderContext
        ctx = BinaryRenderContext(shared_vertices=True)
        result = run(xml, mesh_render_context=ctx)
        
        # Just get the GOM, skip mesh rendering
        result = run(xml, mesh_render_context=False)
        for panel in result.gom.iter_panels():
            print(panel.world_transform)
        
        # With profiling
        result = run(xml, profile=True)
        print(result.timings)
    """
    
    # Build effective config
    if config is None:
        config = GXMLConfig()
    
    # Apply overrides
    if backend is not None:
        config.backend = backend
    if mesh_render_context is not _NOT_PROVIDED:
        config.mesh_render_context = mesh_render_context
    if profile is not None:
        config.profile = profile
    
    # Clear any previous profiling data
    if config.profile:
        reset_profile()
    
    with perf_marker("run"):
        # Set backend
        from elements.solvers import set_solver_backend
        set_solver_backend(config.backend)
        
        # Validate: Check XML against schema
        with perf_marker("validate"):
            validate_xml(xml)
        
        # Parse: GXML string → GOM
        gom = GXMLParser.parse(xml)
        
        # Layout: compute positions, intersections, geometry
        GXMLLayout.layout(gom)
        
        # Render: GOM → mesh
        mesh = None
        if config.mesh_render_context is False:
            render_ctx = None
        elif config.mesh_render_context is None:
            from render_engines.binary_render_context import BinaryRenderContext
            render_ctx = BinaryRenderContext()
        else:
            render_ctx = config.mesh_render_context
        
        if render_ctx is not None:
            GXMLRender.render(gom, render_ctx)
            
            # Get mesh from context
            if hasattr(render_ctx, 'get_output'):
                mesh = render_ctx.get_output()
        
        # Count panels
        def count_panels(element):
            count = 1 if type(element).__name__ == 'GXMLPanel' else 0
            for child in element.children:
                count += count_panels(child)
            return count
        
        stats = {
            'panel_count': count_panels(gom),
            'intersection_count': 0,
            'polygon_count': 0,
        }
        if hasattr(gom, '_panel_solution_cache'):
            intersection_solution, _, _ = gom._panel_solution_cache
            stats['intersection_count'] = len(intersection_solution.intersections)
    
    # Collect timings AFTER run marker ends
    timings = get_profile_results() if config.profile else None
    
    return GXMLResult(
        gom=gom,
        mesh=mesh,
        timings=timings,
        stats=stats,
    )
