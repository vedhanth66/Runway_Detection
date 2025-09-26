import gradio as gr
import numpy as np
import cv2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

# Import our sophisticated pipeline
import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance for optimal performance
DEVICE = "cpu"  # Change to "cuda" for GPU acceleration
ANALYZER = None

def initialize_model():
    """Initialize the model once at startup"""
    global ANALYZER
    try:
        logger.info("🚀 Initializing RunwayNet...")
        ANALYZER = pipeline.load_model(device=DEVICE)
        logger.info("✅ Model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {str(e)}")
        ANALYZER = None

# Advanced visualization functions
def create_performance_radar(results: Dict) -> go.Figure:
    """Create an elegant performance radar chart"""
    
    metrics = {
        'IoU Score': results.get('iou_score', 0),
        'Anchor Score': results.get('anchor_score', 0),
        'Confidence': results.get('confidence', 0),
        'Edge Density': results.get('edge_density', 0),
        'Connectivity': results.get('connectivity_score', 0)
    }
    
    # Create figure with dark theme
    fig = go.Figure()
    
    # Add radar trace
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        name='Performance Metrics',
        line=dict(color='rgba(79, 172, 254, 1)', width=3),
        fillcolor='rgba(79, 172, 254, 0.25)',
        marker=dict(size=8, color='rgba(79, 172, 254, 1)')
    ))
    
    # Style the radar chart
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                tickfont=dict(size=12, color='rgba(255, 255, 255, 0.8)'),
                gridcolor='rgba(255, 255, 255, 0.2)',
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color='white', family='Inter'),
                rotation=90,
                direction='clockwise'
            ),
            bgcolor='rgba(0, 0, 0, 0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white', family='Inter'),
        height=450,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_metrics_dashboard(results: Dict) -> go.Figure:
    """Create a comprehensive metrics dashboard"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Scores', 'Processing Metrics', 
                       'Geometric Analysis', 'Quality Indicators'),
        specs=[[{"type": "bar"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Performance Scores (Bar Chart)
    scores = ['IoU', 'Anchor', 'Mean', 'Confidence']
    values = [
        results.get('iou_score', 0),
        results.get('anchor_score', 0), 
        results.get('mean_score', 0),
        results.get('confidence', 0)
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig.add_trace(
        go.Bar(x=scores, y=values, marker_color=colors, name='Scores',
               text=[f'{v:.3f}' for v in values], textposition='auto'),
        row=1, col=1
    )
    
    # 2. Processing Time Indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=results.get('processing_time', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Processing Time (s)"},
            gauge={
                'axis': {'range': [None, 2]},
                'bar': {'color': "#4ECDC4"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 1.0], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1.5
                }
            }
        ),
        row=1, col=2
    )
    
    # 3. Geometric Analysis (Scatter)
    if 'runway_area' in results and 'aspect_ratio' in results:
        fig.add_trace(
            go.Scatter(
                x=[results.get('runway_area', 0)],
                y=[results.get('aspect_ratio', 1)],
                mode='markers',
                marker=dict(size=20, color='#FF6B6B'),
                name='Runway Geometry',
                text=[f"Area: {results.get('runway_area', 0):.0f}<br>Ratio: {results.get('aspect_ratio', 1):.2f}"],
                hoverinfo='text'
            ),
            row=2, col=1
        )
    
    # 4. Quality Indicators
    quality_metrics = ['Coverage', 'Edge Quality', 'Connectivity']
    quality_values = [
        results.get('coverage_ratio', 0),
        results.get('edge_density', 0),
        results.get('connectivity_score', 0)
    ]
    
    fig.add_trace(
        go.Bar(x=quality_metrics, y=quality_values, 
               marker_color=['#E74C3C', '#F39C12', '#8E44AD'],
               name='Quality', text=[f'{v:.3f}' for v in quality_values],
               textposition='auto'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        title_font=dict(color='white', size=16)
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
    
    return fig

def create_coordinate_visualization(results: Dict) -> go.Figure:
    """Create sophisticated coordinate visualization"""
    
    fig = go.Figure()
    
    # Extract coordinates
    ledg = results.get('ledg_coords', {})
    redg = results.get('redg_coords', {})
    ctl = results.get('ctl_coords', {})
    
    # Draw runway edges
    if ledg.get('start') and ledg.get('end'):
        fig.add_trace(go.Scatter(
            x=[ledg['start'][0], ledg['end'][0]],
            y=[ledg['start'][1], ledg['end'][1]],
            mode='lines+markers',
            name='Left Edge',
            line=dict(color='#E74C3C', width=4),
            marker=dict(size=12, symbol='circle')
        ))
    
    if redg.get('start') and redg.get('end'):
        fig.add_trace(go.Scatter(
            x=[redg['start'][0], redg['end'][0]], 
            y=[redg['start'][1], redg['end'][1]],
            mode='lines+markers',
            name='Right Edge',
            line=dict(color='#3498DB', width=4),
            marker=dict(size=12, symbol='circle')
        ))
    
    if ctl.get('start') and ctl.get('end'):
        fig.add_trace(go.Scatter(
            x=[ctl['start'][0], ctl['end'][0]], 
            y=[ctl['start'][1], ctl['end'][1]],
            mode='lines+markers',
            name='Center Line',
            line=dict(color='#F1C40F', width=3, dash='dot'),
            marker=dict(size=10, symbol='diamond')
        ))
    
    # Style the plot
    fig.update_layout(
        title="Runway Geometry Detection",
        xaxis_title="X Coordinate (pixels)",
        yaxis_title="Y Coordinate (pixels)",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        height=500,
        margin=dict(l=60, r=60, t=60, b=60)
    )
    
    # Invert Y-axis for image coordinates and style axes
    fig.update_yaxes(autorange="reversed", gridcolor='rgba(255,255,255,0.1)', 
                     zerolinecolor='rgba(255,255,255,0.2)')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', 
                     zerolinecolor='rgba(255,255,255,0.2)')
    
    return fig

def format_analysis_report(results: Dict) -> str:
    """Generate a comprehensive analysis report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "✅ PASSED" if results.get('boolean_score', False) else "❌ FAILED"
    
    # Performance summary
    performance_section = f"""
## 🎯 Analysis Summary
**Timestamp:** {timestamp}  
**Status:** {status}  
**Processing Time:** {results.get('processing_time', 0):.3f} seconds

## 📊 Performance Metrics
| Metric | Score | Status |
|--------|-------|--------|
| **IoU Score** | {results.get('iou_score', 0):.4f} | {'🟢' if results.get('iou_score', 0) > 0.8 else '🟡' if results.get('iou_score', 0) > 0.6 else '🔴'} |
| **Anchor Score** | {results.get('anchor_score', 0):.4f} | {'🟢' if results.get('anchor_score', 0) > 0.85 else '🟡' if results.get('anchor_score', 0) > 0.7 else '🔴'} |
| **Confidence** | {results.get('confidence', 0):.4f} | {'🟢' if results.get('confidence', 0) > 0.9 else '🟡' if results.get('confidence', 0) > 0.75 else '🔴'} |
| **Mean Score** | {results.get('mean_score', 0):.4f} | {'🟢' if results.get('mean_score', 0) > 0.8 else '🟡' if results.get('mean_score', 0) > 0.6 else '🔴'} |
"""
    
    # Geometric analysis
    if 'runway_area' in results:
        geometric_section = f"""
## 📐 Geometric Analysis
- **Runway Area:** {results.get('runway_area', 0):,.0f} px²
- **Aspect Ratio:** {results.get('aspect_ratio', 1):.2f}:1
- **Orientation:** {results.get('orientation_angle', 0):.1f}°
- **Coverage Ratio:** {results.get('coverage_ratio', 0):.3f}
"""
    else:
        geometric_section = "\n## 📐 Geometric Analysis\n*No geometric data available*\n"
    
    # Quality indicators
    quality_section = f"""
## 🔍 Quality Indicators
- **Edge Density:** {results.get('edge_density', 0):.4f}
- **Connectivity Score:** {results.get('connectivity_score', 0):.4f}
- **Detection Confidence:** {results.get('confidence', 0)*100:.1f}%
"""
    
    return performance_section + geometric_section + quality_section

def create_status_card(results: Dict) -> str:
    """Create a status card with key metrics"""
    
    status = results.get('boolean_score', False)
    confidence = results.get('confidence', 0)
    processing_time = results.get('processing_time', 0)
    
    status_color = "#2ECC71" if status else "#E74C3C"
    status_text = "DETECTION SUCCESSFUL" if status else "DETECTION FAILED"
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1) 0%, rgba(52, 152, 219, 0.1) 100%);
        border: 2px solid {status_color};
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    ">
        <h2 style="color: {status_color}; margin: 0 0 15px 0; font-family: 'Inter', sans-serif;">
            {'🎯' if status else '⚠️'} {status_text}
        </h2>
        <div style="display: flex; justify-content: space-around; margin: 20px 0;">
            <div style="text-align: center;">
                <div style="font-size: 2em; font-weight: bold; color: #3498DB;">
                    {confidence:.3f}
                </div>
                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9em;">
                    Confidence
                </div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2em; font-weight: bold; color: #E67E22;">
                    {processing_time:.2f}s
                </div>
                <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9em;">
                    Processing Time
                </div>
            </div>
        </div>
    </div>
    """
    
    return card_html

# Main analysis function
def run_comprehensive_analysis(input_image: Optional[np.ndarray]) -> Tuple:
    """
    Execute comprehensive runway analysis with sophisticated error handling
    """
    
    if input_image is None:
        return create_empty_results()
    
    if ANALYZER is None:
        return create_error_results("Model not initialized")
    
    try:
        # Run the analysis pipeline
        results = pipeline.run_full_pipeline(input_image, ANALYZER, device=DEVICE)
        
        # Generate all visualizations and reports
        performance_radar = create_performance_radar(results)
        metrics_dashboard = create_metrics_dashboard(results)
        coordinate_viz = create_coordinate_visualization(results)
        analysis_report = format_analysis_report(results)
        status_card = create_status_card(results)
        
        # Prepare coordinate data for JSON export
        coordinate_data = {
            "runway_geometry": {
                "left_edge": results.get('ledg_coords', {}),
                "right_edge": results.get('redg_coords', {}),
                "center_line": results.get('ctl_coords', {})
            },
            "geometric_properties": {
                "area": results.get('runway_area', 0),
                "aspect_ratio": results.get('aspect_ratio', 1),
                "orientation_angle": results.get('orientation_angle', 0)
            },
            "performance_metrics": {
                "iou_score": results.get('iou_score', 0),
                "anchor_score": results.get('anchor_score', 0),
                "confidence": results.get('confidence', 0),
                "processing_time": results.get('processing_time', 0)
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": DEVICE,
                "status": "success" if results.get('boolean_score', False) else "warning"
            }
        }
        
        # Determine visibility of status indicators
        show_success = results.get('boolean_score', False)
        show_warning = not show_success
        
        logger.info(f"✅ Analysis completed successfully in {results.get('processing_time', 0):.3f}s")
        
        return (
            results.get('visual_result', input_image),  # Processed image
            status_card,  # Status card HTML
            analysis_report,  # Detailed report
            performance_radar,  # Performance radar chart
            metrics_dashboard,  # Comprehensive metrics dashboard
            coordinate_viz,  # Coordinate visualization
            json.dumps(coordinate_data, indent=2),  # JSON data
            gr.update(visible=show_success),  # Success indicator
            gr.update(visible=show_warning)   # Warning indicator
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return create_error_results(f"Analysis failed: {str(e)}")

def create_empty_results() -> Tuple:
    """Create empty results for when no image is provided"""
    
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Upload an image to start analysis",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.5)'),
        height=400
    )
    
    empty_status = """
    <div style="
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    ">
        <h3 style="color: rgba(255, 255, 255, 0.7); margin: 0;">
            📤 Ready for Analysis
        </h3>
        <p style="color: rgba(255, 255, 255, 0.5); margin: 15px 0 0 0;">
            Upload a runway image to begin comprehensive detection and analysis
        </p>
    </div>
    """
    
    return (
        None,  # No processed image
        empty_status,  # Empty status card
        "## 📋 Analysis Report\n*Awaiting image upload...*",  # Empty report
        empty_fig,  # Empty radar chart
        empty_fig,  # Empty dashboard
        empty_fig,  # Empty coordinate viz
        "{}",  # Empty JSON
        gr.update(visible=False),  # Hide success
        gr.update(visible=False)   # Hide warning
    )

def create_error_results(error_message: str) -> Tuple:
    """Create error results when analysis fails"""
    
    error_fig = go.Figure()
    error_fig.update_layout(
        title=f"Error: {error_message}",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E74C3C'),
        height=400
    )
    
    error_status = f"""
    <div style="
        background: rgba(231, 76, 60, 0.1);
        border: 2px solid #E74C3C;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
    ">
        <h3 style="color: #E74C3C; margin: 0;">
            ❌ Analysis Error
        </h3>
        <p style="color: rgba(255, 255, 255, 0.8); margin: 15px 0 0 0;">
            {error_message}
        </p>
    </div>
    """
    
    return (
        None,
        error_status,
        f"## ❌ Error Report\n**Error:** {error_message}",
        error_fig,
        error_fig,
        error_fig,
        "{}",
        gr.update(visible=False),
        gr.update(visible=True)
    )

# Enhanced UI styling
CUSTOM_CSS = """
/* Modern glass morphism design */
.gradio-container {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
    min-height: 100vh;
}

/* Enhanced glass effect for components */
.block {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* Button styling */
.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
}

/* Tab styling */
.tab-nav {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 15px !important;
    padding: 5px !important;
}

.selected {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Animation classes */
.fade-in {
    animation: fadeIn 0.5s ease-in !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Status indicator animations */
.pulse-success {
    animation: pulseSuccess 2s infinite !important;
}

@keyframes pulseSuccess {
    0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
    100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
}

/* Typography */
h1, h2, h3 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
"""

# Initialize model at startup
initialize_model()

# Create the elegant Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
    title="RunwayNet - Advanced Detection System",
    analytics_enabled=False
) as demo:
    
    # Header section with modern styling
    gr.HTML("""
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 25px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
    ">
        <h1 style="
            color: white;
            font-size: 3.5em;
            margin: 0;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        ">
            ✈️ RunwayNet
        </h1>
        <p style="
            color: rgba(255,255,255,0.8);
            font-size: 1.3em;
            margin: 20px 0 0 0;
            font-weight: 300;
        ">
            Advanced AI-Powered Runway Detection & Analysis Platform
        </p>
        <div style="
            margin-top: 20px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 25px;
            display: inline-block;
        ">
            <span style="color: #2ECC71; font-weight: 600;">🟢 System Online</span>
            <span style="color: rgba(255,255,255,0.6); margin-left: 20px;">
                Device: {DEVICE.upper()}
            </span>
        </div>
    </div>
    """.replace("{DEVICE.upper()}", DEVICE.upper()))
    
    # Main interface layout
    with gr.Row(equal_height=True):
        
        # Left panel - Input and controls
        with gr.Column(scale=2, elem_classes=["fade-in"]):
            gr.HTML("""
            <div style="
                background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
                color: white;
                padding: 20px;
                border-radius: 15px 15px 0 0;
                text-align: center;
                font-weight: 600;
                font-size: 1.1em;
            ">
                📤 Image Upload & Analysis
            </div>
            """)
            
            with gr.Group():
                input_image = gr.Image(
                    type="numpy",
                    label=None,
                    height=450,
                    container=False,
                    elem_classes=["fade-in"]
                )
                
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🚀 Run Complete Analysis",
                        variant="primary",
                        size="lg",
                        scale=2
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="lg",
                        scale=1
                    )
        
        # Right panel - Results and analytics
        with gr.Column(scale=3, elem_classes=["fade-in"]):
            gr.HTML("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px 15px 0 0;
                text-align: center;
                font-weight: 600;
                font-size: 1.1em;
            ">
                📊 Analysis Results & Insights
            </div>
            """)
            
            # Status card at the top
            status_display = gr.HTML(label=None, container=False)
            
            # Tabbed results interface
            with gr.Tabs():
                
                with gr.TabItem("🖼️ Visual Detection", elem_id="visual-tab"):
                    output_image = gr.Image(
                        type="numpy",
                        label=None,
                        height=450,
                        container=False,
                        elem_classes=["fade-in"]
                    )
                
                with gr.TabItem("📋 Analysis Report", elem_id="report-tab"):
                    analysis_report = gr.Markdown(
                        value="## 📋 Analysis Report\n*Ready for analysis...*",
                        container=False,
                        elem_classes=["fade-in"]
                    )
                
                with gr.TabItem("🎯 Performance Radar", elem_id="radar-tab"):
                    performance_radar = gr.Plot(
                        label=None,
                        container=False,
                        elem_classes=["fade-in"]
                    )
                
                with gr.TabItem("📈 Metrics Dashboard", elem_id="metrics-tab"):
                    metrics_dashboard = gr.Plot(
                        label=None,
                        container=False,
                        elem_classes=["fade-in"]
                    )
                
                with gr.TabItem("📐 Geometry Viz", elem_id="geometry-tab"):
                    coordinate_plot = gr.Plot(
                        label=None,
                        container=False,
                        elem_classes=["fade-in"]
                    )
                
                with gr.TabItem("💾 Export Data", elem_id="data-tab"):
                    json_output = gr.Code(
                        language="json",
                        label="Complete Analysis Data (JSON)",
                        container=False,
                        elem_classes=["fade-in"]
                    )
    
    # Status indicators with modern styling
    with gr.Row(visible=False) as success_row:
        gr.HTML("""
        <div style="
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.2) 0%, rgba(39, 174, 96, 0.2) 100%);
            border: 2px solid #2ECC71;
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            margin: 20px 0;
            backdrop-filter: blur(15px);
        " class="pulse-success">
            <h3 style="color: #2ECC71; margin: 0; font-size: 1.5em;">
                🎉 Analysis Complete!
            </h3>
            <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0;">
                Runway detection and analysis completed successfully
            </p>
        </div>
        """)
    
    with gr.Row(visible=False) as warning_row:
        gr.HTML("""
        <div style="
            background: linear-gradient(135deg, rgba(231, 76, 60, 0.2) 0%, rgba(192, 57, 43, 0.2) 100%);
            border: 2px solid #E74C3C;
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            margin: 20px 0;
            backdrop-filter: blur(15px);
        ">
            <h3 style="color: #E74C3C; margin: 0; font-size: 1.5em;">
                ⚠️ Analysis Warning
            </h3>
            <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0;">
                Detection completed with low confidence scores
            </p>
        </div>
        """)
    
    # Footer with system information
    gr.HTML(f"""
    <div style="
        text-align: center;
        padding: 30px;
        margin-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.02);
        border-radius: 0 0 25px 25px;
    ">
        <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.9em;">
            🏛️ <strong>RunwayNet v3.0</strong> | 
            Powered by Advanced Computer Vision | 
            Runtime: {DEVICE.upper()} | 
            <span style="color: #4ECDC4;">
                <a href="#" style="color: #4ECDC4; text-decoration: none;">Documentation</a>
            </span> | 
            <span style="color: #4ECDC4;">
                <a href="#" style="color: #4ECDC4; text-decoration: none;">Support</a>
            </span>
        </p>
        <p style="color: rgba(255,255,255,0.4); margin: 10px 0 0 0; font-size: 0.8em;">
            Built with ❤️ for aviation safety and precision landing systems
        </p>
    </div>
    """)
    
    # Event handlers
    analyze_btn.click(
        fn=run_comprehensive_analysis,
        inputs=[input_image],
        outputs=[
            output_image,
            status_display,
            analysis_report,
            performance_radar,
            metrics_dashboard,
            coordinate_plot,
            json_output,
            success_row,
            warning_row
        ],
        show_progress=True
    )
    
    clear_btn.click(
        fn=lambda: create_empty_results(),
        outputs=[
            input_image,
            status_display,
            analysis_report,
            performance_radar,
            metrics_dashboard,
            coordinate_plot,
            json_output,
            success_row,
            warning_row
        ]
    )

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        quiet=False
    )