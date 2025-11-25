### Derive from the base container image
FROM dockerdex.umcn.nl:5005/diag/base-images:base-pt2.7.1

### Define a variable for the code source directory
ARG CODE_DIR="/home/filipwinzell/temporary/"

### Copy your code to the container image
COPY "." ${CODE_DIR}

### Install python packages
RUN pip3 install torch-summary
RUN pip3 install perceiver-pytorch
RUN pip3 install tensorboard
RUN pip3 install openpyxl
RUN pip3 install timm
RUN pip3 install scikit-survival
RUN pip3 install torchio

