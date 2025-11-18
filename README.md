This repo contains code used in the following AAAI26 paper:

Jones, S. J., Wray, R. E., & Laird, J. E. (2026). Requirements for Aligned, Dynamic Resolution of Conflicts in Operational Constraints. _Proceedings of the 40th Annual AAAI Conference on Artificial Intelligence_. Singapore. 

A preprint of the paper is available on arXiv: https://doi.org/10.48550/arXiv.2511.10952 along with a Technical Appendix to the conference paper.

The paper introduces describe Online, Aligned Mitigation of Novel Constraint Conflicts (OAMNCC). We argue OAMNCC is a key capability required for long-lived, autonomous agents. The paper presents a number of illustrative simulation use cases to highlight the role of various kinds of knowledge required for OAMNCC. This repository includes the code for these simulations and the data analysis scripts.

There are two primary scenarios (with variations as described in the paper):

1. The sailor overboard scenario is meant to be run as a jupyter notebook.
2. The piracy interdiction scenarios are just run as python scripts. 

- For both scenarios, parameters can be adjusted to simulate various ship and scenarios properties, but the values in the code here (i.e., the ones committed in the respository) are the ones used in the final paper.
