# Start with BioSim base image.
ARG BASE_IMAGE=latest
FROM ghcr.io/ccpbiosim/jupyterhub-base:$BASE_IMAGE

LABEL maintainer="James Gebbie-Rayet <james.gebbie@stfc.ac.uk>"

# Switch to jovyan user.
USER $NB_USER
WORKDIR $HOME

# Install workshop deps
RUN mamba install -y -c conda-forge \
    conda-forge::openff-toolkit-examples=0.19.0 \
    conda-forge::openff-interchange=0.5.4 \
    conda-forge::ambertools=26.3 \
    conda-forge::MDAnalysis=2.10.0 \
    conda-forge::prolif=2.2.1 \
    conda-forge::py3dmol=2.5.4 \
    conda-forge::openbabel=3.2.1

#COPY docker/fix-nglview.sh /tmp/fix-nglview.sh
#RUN bash /tmp/fix-nglview.sh

# Get workshop files and move them to jovyan directory.
COPY --chown=1000:100 . .
RUN rm -rf .gitignore .pre-commit-config.yaml environment.yaml LICENSE Makefile README.md devtools docker notebooks-rendered .git .github

# Copy updated lab workspace
#COPY --chown=1000:100 docker/default-37a8.jupyterlab-workspace /home/jovyan/.jupyter/lab/workspaces/default-37a8.jupyterlab-workspace


# Always finish with non-root user as a precaution.
USER $NB_USER
