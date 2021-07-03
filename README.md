# CRNs
GPU-accelerated Gillespie simulations for stochastic chemical reaction networks

Depends on Python 3, numpy, PyTorch, tqdm.

tssa.py implements a GPU-accellerated stochastic_reaction_network class, which is able to sample many processesses in parallel from the chemical master equation with time-varying external drives, as long as these drives change only at a discrete set of timepoints. 

network_constructurs.py implements functions to create the chemical reaction networks studied in Ueltzhöffer et al. (2021).

simple_drive_network.py implements the simple_reaction_network class, which simulates deterministic rate equations for chemical reaction networks. It uses an interface very similar to stochastic_reaction_network, but it only relies on numpy and does not feature GPU acceleration.

To see how these classes are used, and how the graphs in Ueltzhöffer et al. were created, see the following scripts:

- create_graphs_for_figure_1b.py
- create_graphs_for_figure_2.py
- create_graphs_for_figure_4b.py
- create_graphs_for_figures_4a_and_4c.py

To run all scripts and create all graphs, you can just run 

```
bash run_all.sh
```

in terminal.
