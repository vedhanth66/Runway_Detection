import gradio as gr
import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import textwrap
import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = pipeline.load_model(device=device)

def create_performance_chart(results):
    metrics = ['IoU Score', 'Anchor Score', 'Confidence', 'Mean Score']
    values = [results["iou_score"], results["anchor_score"], results["confidence"], results["mean_score"]]
    fig = go.Figure(data=[go.Scatterpolar(r=values, theta=metrics, fill='toself', name='metrics', line=dict(color='#4ECDC4'))])
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400)
    return fig

def create_coordinate_visualization(results):
    fig = go.Figure()
    if results["ledg_coords"]:
        ledg, redg, ctl = results["ledg_coords"], results["redg_coords"], results["ctl_coords"]
        fig.add_trace(go.Scatter(x=[ledg["start"][0], ledg["end"][0]], y=[ledg["start"][1], ledg["end"][1]], mode='lines+markers', name='left_edge', line=dict(color='#FF6B6B', width=3)))
        fig.add_trace(go.Scatter(x=[redg["start"][0], redg["end"][0]], y=[redg["start"][1], redg["end"][1]], mode='lines+markers', name='right_edge', line=dict(color='#45B7D1', width=3)))
        fig.add_trace(go.Scatter(x=[ctl["start"][0], ctl["end"][0]], y=[ctl["start"][1], ctl["end"][1]], mode='lines+markers', name='center_line', line=dict(color='yellow', width=2, dash='dot')))
    fig.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=400, yaxis=dict(autorange="reversed"))
    return fig

def format_results_summary(results):
    summary = f"""
    ### Performance Metrics
    - **IoU Score:** {results['iou_score']:.4f} (N/A for new images)
    - **Anchor Score:** {results['anchor_score']:.4f}
    - **Mean Score:** {results['mean_score']:.4f}
    - **Confidence:** {results['confidence']:.4f}
    - **Processing Time:** {results['processing_time']:.2f}s
    ### Detection Status
    **Overall Result:** {'PASS' if results['boolean_score'] else 'FAIL - Orientation Invalid'}
    """
    return textwrap.dedent(summary)

def run_analysis(input_image):
    if input_image is None:
        empty_fig = go.Figure().update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        return (None, "## awaiting analysis...", empty_fig, empty_fig, "{}", gr.update(visible=False), gr.update(visible=False))
    
    results = pipeline.run_full_pipeline(input_image, model, device=device)
    if results is None:
        return
    
    performance_chart = create_performance_chart(results)
    coordinate_viz = create_coordinate_visualization(results)
    summary_text = format_results_summary(results)
    
    coordinate_data = {
        "left_edge": results["ledg_coords"], 
        "right_edge": results["redg_coords"], 
        "center_line": results["ctl_coords"],
        "metadata": {
            "timestamp": datetime.now().isoformat(), 
            "processing_time": f"{results['processing_time']:.2f}s", 
            "confidence": f"{results['confidence']:.4f}"
        }
    }
    
    show_warning = not results['boolean_score']
    
    return (
        results["visual_result"], summary_text, performance_chart, coordinate_viz,
        json.dumps(coordinate_data, indent=2), gr.update(visible=True), gr.update(visible=show_warning)
    )

with gr.Blocks(theme=gr.themes.Glass(), title="RunwayNet Advanced Dashboard") as demo:
    gr.HTML("""<div style="text-align: center; padding: 20px;"><h1 style="color: white; font-size: 2.5em;">RunwayNet Advanced Dashboard</h1></div>""")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(type="numpy", label="Upload Runway Image", height=400)
            with gr.Row():
                analyze_btn = gr.Button("Run Complete Analysis", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary", size="lg")
        
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Visual Detection"):
                    output_image = gr.Image(type="numpy", label="Detection Result", height=400)
                with gr.TabItem("Detailed Report"):
                    summary_output = gr.Markdown(label="Analysis Summary")
                with gr.TabItem("Performance Analytics"):
                    performance_radar = gr.Plot(label="Performance Radar Chart")
                with gr.TabItem("Geometric Analysis"):
                    coordinate_plot = gr.Plot(label="Coordinate Visualization")
                with gr.TabItem("Raw Data"):
                    json_output = gr.Code(language="json", label="Coordinate Data (JSON)")
    
    with gr.Row(visible=False) as progress_row:
        gr.HTML("""<div style='text-align: center; padding: 20px; background: rgba(0,255,150,0.2); border-radius: 10px; border: 1px solid #4ECDC4;'><h3 style='color: #4ECDC4;'>Analysis Complete!</h3></div>""")
    with gr.Row(visible=False) as warning_row:
        gr.HTML("""<div style='text-align: center; padding: 20px; background: rgba(255,165,0,0.2); border-radius: 10px; border: 1px solid orange;'><h3 style='color: orange;'>Low Score Warning</h3></div>""")

    analyze_btn.click(
        fn=run_analysis, inputs=[input_image],
        outputs=[output_image, summary_output, performance_radar, coordinate_plot, json_output, progress_row, warning_row]
    )
    
    clear_btn.click(
        lambda: (None, "## awaiting analysis...", None, None, "{}", gr.update(visible=False), gr.update(visible=False)),
        outputs=[input_image, summary_output, performance_radar, coordinate_plot, json_output, progress_row, warning_row]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
