"""
GXML Engine - Main entry point for processing GXML content.

This module provides the primary API for parsing, laying out, and rendering
GXML (Geometric XML) content.

Usage:
    from gxml_engine import run, GXMLConfig
    
    # Simple usage with defaults
    result = run(xml_string)
    
    # With configuration
    config = GXMLConfig(backend='c', output_format='json')
    result = run(xml_string, config=config)
    
    # With individual options
    result = run(xml_string, backend='c', output_format='json')
    
    # With profiling - shows timing for all instrumented code sections
    result = run(xml_string, profile=True)
    print(result.timings)
    # {'parse': {'count': 1, 'total_ms': 1.2, ...}, 
    #  'on_post_layout': {'count': 200, 'total_ms': 650.0, ...}, ...}
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Literal, Dict, Union
import time

from gxml_parser import GXMLParser
from gxml_layout import GXMLLayout
from gxml_render import GXMLRender
from gxml_profile import (
    enable_profiling, 
    reset_profile, 
    get_profile_results,
    push_perf_marker,
    pop_perf_marker,
)


# =============================================================================
# Configuration
# =============================================================================

Backend = Literal['cpu', 'c', 'taichi', 'gpu']
OutputFormat = Literal['json', 'binary', 'indexed', 'dict', 'none']


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
        
        output_format: How to output the rendered geometry.
            - 'json': Return JSON-serializable dict with full panel metadata
            - 'binary': Return packed binary data (for WebGL/Three.js) - per-panel polygons
            - 'indexed': Return indexed mesh data (GXMF format) - shared vertices + triangle indices
            - 'dict': Return minimal geometry dict (simple format)
            - 'none': Skip rendering, return parsed/laid-out tree only
        
        profile: Enable timing profiling of pipeline stages.
            When True, returns timing data from all instrumented code sections.
            Markers are placed throughout the codebase using push_perf_marker/pop_perf_marker.
        
        validate: Validate XML structure before processing.
        
        render_context: Custom render context instance (overrides output_format).
    """
    backend: Backend = 'cpu'
    output_format: OutputFormat = 'dict'
    profile: bool = False
    validate: bool = False
    render_context: Optional[Any] = None
    
    # Additional options that can be expanded
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GXMLResult:
    """
    Result from running the GXML pipeline.
    
    Attributes:
        root: The parsed and laid-out GXML element tree.
        output: The rendered output (format depends on config.output_format).
        timings: Timing data from profiled code sections (if config.profile=True).
            Each key is a marker name, value contains count, total_ms, avg_ms, min_ms, max_ms.
        stats: Statistics about the processed content.
    """
    root: Any
    output: Any = None
    timings: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, int]] = None
    
    def get_web_timings(self) -> Dict[str, float]:
        """
        Get timing data formatted for web responses.
        
        Returns a flat dict with timing values in milliseconds, suitable for
        HTTP headers or JSON responses. Maps internal marker names to 
        web-friendly names.
        
        Returns:
            Dict with keys: parse, measure, prelayout, layout, postlayout,
            render, intersection, face, geometry, fastmesh (all in ms)
        """
        return format_timings_for_web(self.timings)


def format_timings_for_web(profile_results: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Format profile results into web-friendly timing dict.
    
    This is the shared timing extraction logic used by both app.py (FastAPI)
    and gxml_server.py (Electron IPC).
    
    Args:
        profile_results: Raw timing dict from get_profile_results()
        
    Returns:
        Dict with web-friendly timing names and values in milliseconds
    """
    if not profile_results:
        return {}
    
    def ms(name: str) -> float:
        return profile_results.get(name, {}).get('total_ms', 0.0)
    
    return {
        'parse': ms('parse'),
        'measure': ms('measure_pass'),
        'prelayout': ms('pre_layout_pass'),
        'layout': ms('layout_pass'),
        'postlayout': ms('post_layout_pass'),
        'render': ms('render'),
        # Solver breakdown (nested within post-layout, or standalone for indexed)
        'intersection': ms('intersection_solver'),
        'face': ms('face_solver'),
        'geometry': ms('geometry_builder'),
        # FastMeshBuilder (indexed pipeline only)
        'fastmesh': ms('fast_mesh_builder'),
        'serialize': ms('serialize'),
    }


# =============================================================================
# Internal Helpers
# =============================================================================

def _set_backend(backend: Backend) -> None:
    """Set the solver backend."""
    from elements.solvers import set_solver_backend
    set_solver_backend(backend)


def _get_render_context(config: GXMLConfig, node: Any = None):
    """Get or create the appropriate render context."""
    # Custom context takes priority
    if config.render_context is not None:
        return config.render_context
    
    if config.output_format == 'houdini':
        # Houdini render context is provided by gxml-houdini plugin
        raise ValueError(
            "output_format='houdini' requires gxml-houdini plugin. "
            "Pass a custom render_context instead."
        )
    
    elif config.output_format == 'json':
        from render_engines.json_render_context import JSONRenderContext
        return JSONRenderContext()
    
    elif config.output_format == 'binary':
        from render_engines.binary_render_context import BinaryRenderContext
        return BinaryRenderContext()
    
    elif config.output_format == 'dict':
        from render_engines.dict_render_context import DictRenderContext
        return DictRenderContext()
    
    elif config.output_format == 'indexed':
        from render_engines.indexed_render_context import IndexedRenderContext
        return IndexedRenderContext()
    
    elif config.output_format == 'none':
        return None
    
    else:
        raise ValueError(f"Unknown output_format: {config.output_format}")


def _collect_stats(root) -> Dict[str, int]:
    """Collect statistics from the processed tree."""
    stats = {
        'panel_count': 0,
        'intersection_count': 0,
        'polygon_count': 0,
    }
    
    # Count panels
    def count_panels(element):
        count = 1 if type(element).__name__ == 'GXMLPanel' else 0
        for child in element.children:
            count += count_panels(child)
        return count
    
    stats['panel_count'] = count_panels(root)
    
    # Get intersection count from cache
    if hasattr(root, '_panel_solution_cache'):
        intersection_solution, _, _ = root._panel_solution_cache
        stats['intersection_count'] = len(intersection_solution.intersections)
    
    return stats


# =============================================================================
# Main API
# =============================================================================

def run(
    xml: str,
    config: Optional[GXMLConfig] = None,
    node: Any = None,
    *,
    # Convenience kwargs that override config
    backend: Optional[Backend] = None,
    output_format: Optional[OutputFormat] = None,
    profile: Optional[bool] = None,
) -> GXMLResult:
    """
    Run the GXML processing pipeline.
    
    This is the main entry point for processing GXML content. It handles:
    1. Parsing XML into element tree
    2. Layout computation (positioning, intersections, geometry)
    3. Rendering to the specified output format
    
    Args:
        xml: The GXML string to process.
        config: Configuration options (GXMLConfig instance).
        node: Houdini node for rendering (required if output_format='houdini').
        backend: Override config.backend.
        output_format: Override config.output_format.
        profile: Override config.profile.
    
    Returns:
        GXMLResult containing the processed tree and output.
        
        When profile=True, result.timings contains timing data from all 
        instrumented code sections (those using push_perf_marker/pop_perf_marker).
        Each marker has: count, total_ms, avg_ms, min_ms, max_ms
    
    Examples:
        # Basic usage
        result = run('<root><panel/></root>')
        
        # With backend selection
        result = run(xml, backend='c')
        
        # With full config
        config = GXMLConfig(backend='c', output_format='json', profile=True)
        result = run(xml, config=config)
        
        # With profiling
        result = run(xml, profile=True)
        for name, stats in result.timings.items():
            print(f"{name}: {stats['total_ms']:.1f}ms ({stats['count']} calls)")
    """
    # Build effective config
    if config is None:
        config = GXMLConfig()
    
    # Apply overrides
    if backend is not None:
        config.backend = backend
    if output_format is not None:
        config.output_format = output_format
    if profile is not None:
        config.profile = profile
    
    # Setup profiling
    if config.profile:
        reset_profile()
        enable_profiling(True)
    
    try:
        # Set backend
        _set_backend(config.backend)
        
        # Parse
        push_perf_marker("parse")
        root = GXMLParser.parse(xml)
        pop_perf_marker()
        
        # Standard pipeline: full layout + render
        push_perf_marker("layout")
        GXMLLayout.layout(root)
        pop_perf_marker()
        
        # Render
        output = None
        render_ctx = _get_render_context(config, node)
        
        if render_ctx is not None:
            push_perf_marker("render")
            GXMLRender.render(root, render_ctx)
            pop_perf_marker()
            
            # Get output from context
            if hasattr(render_ctx, 'get_output'):
                output = render_ctx.get_output()
        
        # Collect timings and stats
        timings = get_profile_results() if config.profile else None
        stats = _collect_stats(root)
        
        return GXMLResult(
            root=root,
            output=output,
            timings=timings,
            stats=stats,
        )
    
    finally:
        # Always disable profiling when done
        if config.profile:
            enable_profiling(False)
