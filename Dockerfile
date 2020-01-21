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
RUN \
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
&& \
  pip3 install --upgrade \
    breathe \
    cloud_sptheme \
    sphinx \
    sphinx-rtd-theme \
    sphinx_autodoc_typehints \
    sphinx_sitemap \
    sphinxcontrib-bibtex
