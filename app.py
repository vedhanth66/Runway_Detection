import gradio as gr
import plotly.express as px
import pandas as pd
import numpy as np
import json

def create_empty_results():
    return (
        None,
        "Status: Awaiting Input",
        "No report generated.",
        px.scatter(),
        px.bar(),
        px.scatter(),
        "{}",
        "Success: None",
        "Warning: None"
    )

def analyze_runway(image):
    df = pd.DataFrame({
        "metric": ["Accuracy", "Precision", "Recall"],
        "value": [0.92, 0.89, 0.94]
    })

    radar = px.line_polar(df, r="value", theta="metric", line_close=True, range_r=[0, 1])
    metrics_chart = px.bar(df, x="metric", y="value", range_y=[0, 1])
    coords = pd.DataFrame({"x": np.random.rand(5), "y": np.random.rand(5)})
    coord_plot = px.scatter(coords, x="x", y="y")

    return (
        image,
        "Status: Analysis Complete",
        "Runway detected successfully with stable confidence levels.",
        radar,
        metrics_chart,
        coord_plot,
        json.dumps(df.to_dict(), indent=2),
        "Success: Runway detected",
        "Warning: Minor anomalies at edges"
    )

with gr.Blocks(title="Runway Analysis Tool") as demo:
    gr.Markdown("## Runway Analysis Dashboard")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Runway Image", type="filepath")
            analyze_btn = gr.Button("Analyze Runway")
            clear_btn = gr.Button("Clear Results")

        with gr.Column(scale=2):
            status_display = gr.Textbox(label="System Status", value="Status: Awaiting Input")
            analysis_report = gr.Textbox(label="Detailed Report")

            with gr.Row():
                performance_radar = gr.Plot(label="Performance Radar")
                metrics_dashboard = gr.Plot(label="Metrics Dashboard")

            coordinate_plot = gr.Plot(label="Runway Coordinate Plot")
            json_output = gr.JSON(label="Raw JSON Output")

            with gr.Row():
                success_row = gr.Textbox(label="Success Message", value="Success: None")
                warning_row = gr.Textbox(label="Warning Message", value="Warning: None")

    all_outputs = [
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

    analyze_btn.click(fn=analyze_runway, inputs=[input_image], outputs=all_outputs)
    clear_btn.click(fn=lambda: create_empty_results(), outputs=all_outputs)

if __name__ == "__main__":
    demo.launch()
