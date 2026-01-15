# CCPBioSim OpenForcefield Workshop

[![ci](https://github.com/ccpbiosim/openff-workshop/actions/workflows/build.yaml/badge.svg?branch=main)](https://github.com/ccpbiosim/openff-workshop/actions/workflows/build.yaml)
[![latest](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fccpbiosim.github.io%2Fworkshop.json&query=%24.containers.openff-workshop.latest&labelColor=grey&logo=github&logoColor=white&label=latest&color=purple)](https://github.com/ccpbiosim/openff-workshop/pkgs/container/openff-workshop) 
[![issues](https://img.shields.io/github/issues/ccpbiosim/openff-workshop?logo=github&labelColor=grey)](https://github.com/CCPBioSim/openff-workshop/issues)
[![pr](https://img.shields.io/github/issues-pr/ccpbiosim/openff-workshop?logo=github&labelColor=grey)](https://github.com/CCPBioSim/openff-workshop/pulls)

This workshop source repository contains the build recipe for a docker container derived from the CCPBioSim JupyterHub image. This container adds the necessary software packages and notebook content to form a deployable course container.

### Materials

We recommend you view the materials in the following order:

* Talk: [Intro to OpenFF](talk-cole-openFFintro.pdf)
* Notebook: [Parameterising small molecules with OpenFF](notebooks/small_molecule_parameterisation.ipynb)
* Notebook: [Parameterisation, molecular dynamics, and basic trajectory analysis for a protein-ligand complex](notebooks/protein_ligand_complex_parameterisation_and_md.ipynb)

Answers to most exercises are given in the [notebooks_with_solutions directory](notebooks_with_solutions).


### More resources

* Main [OpenFF docs](https://docs.openforcefield.org/en/latest/)
  * See "Projects" on the left for package-specific documentation
  * Ecosystem-wide [examples](https://docs.openforcefield.org/en/latest/examples.html)
* [SMIRNOFF](https://openforcefield.github.io/standards/standards/smirnoff/) specification
* [Discussions](https://github.com/orgs/openforcefield/discussions) - for general usage questions


### Acknowledgements

Most of the material for the notebook [Parameterising small molecules with OpenFF](notebooks/small_molecule_parameterisation.ipynb) was adapted from the [2023 CCPBioSim Workshop Open Force Field Sessions](https://github.com/openforcefield/ccpbiosim-2023?) created by Matt Thompson and Jeff Wagner.

Most of the material for the notebook [Parameterisation, molecular dynamics, and basic trajectory analysis for a protein-ligand complex](notebooks/protein_ligand_complex_parameterisation_and_md.ipynb) was adapted from the OpenFF [toolkit showcase](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-toolkit/toolkit_showcase/toolkit_showcase.html) and the [ProLIF Ligand-protein MD tutorial](https://prolif.readthedocs.io/en/latest/notebooks/md-ligand-protein.html#ligand-protein-md).

## How to Use

This training course is deployed on the [CCPBioSim](www.ccpbiosim.ac.uk) website via our cloud infrastructure, however you can deploy on your own machine with docker.

Pull the container from our repository::

    docker pull ghcr.io/ccpbiosim/openff-workshop:latest

In our containers we are using the JupyterHub default port 8888, so you should
forward this port when deploying locally::

    docker run -p 8888:8888 ghcr.io/ccpbiosim/openff-workshop:latest

## Authors

Workshop Content Authors:

- Danny Cole
- Finlay Clark

## Contact

Please direct all questions and feedback to [Finlay Clark](mailto:charles.laughton@nottingham.ac.uk)
