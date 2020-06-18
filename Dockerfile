# Base image
FROM python:3.6

# 1. Base packages
# 2. Packages for testing
# 3. Packages needed for icet
# 4. Packages for setting up documentation
RUN \
  apt-get update -qy && \
  apt-get upgrade -qy && \
  apt-get install -qy \
    doxygen \
    graphviz \
    zip
# Packages for testing
# Packages needed for icet
# Packages for building documentation
RUN \
  pip3 install --upgrade \
    pip \
  && \
  pip3 install --upgrade \
    coverage \
    flake8 \
    mypy \
  && \
  pip3 install --upgrade \
    ase \
    mip \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    spglib \
    xdoctest \
  && \
  pip3 install --upgrade \
    breathe \
    cloud_sptheme \
    sphinx \
    sphinx-rtd-theme \
    sphinx_sitemap \
    sphinxcontrib-bibtex
