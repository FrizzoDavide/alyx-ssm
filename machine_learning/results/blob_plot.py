"""
Python script to produce the blob plot of num parameters vs FLOPs vs sample accuracies
"""

import argparse
import ipdb
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
ml_path = os.path.dirname(script_dir)

sys.path.append(ml_path)

from src.utils import utils

parser = argparse.ArgumentParser(description="Blob Plot Experiment")

parser.add_argument(
    "--show_plot",
    action="store_true",
    help="If set, show the plot to the screen"
)

parser.add_argument(
    "--save_plot",
    action="store_true",
    help="If set, save the plot in a png file"
)

parser.add_argument(
    "--remove_mlp",
    action="store_true",
    help="If set, remove the MLP data from the plot"
)

args = parser.parse_args()

plot_path = os.path.join(ml_path,"plots")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

blob_data_path = os.path.join(os.getcwd(),"blob_plot_data.json")

with open(blob_data_path,"r") as f:
    blob_dict = json.load(f)

blob_df = pd.DataFrame(blob_dict)

blob_df = blob_df.rename(
    columns = {
        "params":"Parameters (M)",
        "accs":"Sample Accuracy",
        "flops": "FLOPs"
    }
)

if args.remove_mlp:
    blob_df = blob_df.loc[1:,:]

fig = px.scatter(
    data_frame=blob_df,
    x="Parameters (M)",
    y="Sample Accuracy",
    color="FLOPs",
    size="FLOPs",
    hover_name="model_name",
    log_x=True,
    size_max=60,
    color_continuous_scale="Viridis",
    text="model_name",
    opacity=0.9,
)
fig.update_traces(
    textposition="bottom center",
    textfont=dict(size=10),
)

if args.show_plot:
    fig.show()

if args.save_plot:
    if args.remove_mlp:
        filename = f"{utils.get_current_time()}_blob_plot_no_mlp.png"
    else:
        filename = f"{utils.get_current_time()}_blob_plot_with_mlp.png"
    plot_path = os.path.join(plot_path, filename)
    fig.write_image(plot_path, scale=1)
    print("-" * 50)
    print(f"Plot saved at: {plot_path}")
    print("-" * 50)
