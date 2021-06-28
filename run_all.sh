#!/bin/bash
mkdir results
mkdir figures

python create_graphs_for_figure_1b.py
python create_graphs_for_figure_2.py
python create_graphs_for_figures_4a_and_4c.py
python create_graphs_for_figure_4b.py

