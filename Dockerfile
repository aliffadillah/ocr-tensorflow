# Gunakan base image Python
FROM python:3.9-slim

# Set lingkungan kerja di container
WORKDIR /app

# Salin file proyek ke dalam container
COPY . .

# Update package manager dan install dependensi sistem, termasuk wget
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential protobuf-compiler wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Tambahkan Miniconda ke PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# Buat dan aktifkan lingkungan Conda
RUN conda create -n ml_bangkit python=3.10.15 -y && \
    conda clean --all -y
ENV PATH="/opt/miniconda/envs/ml_bangkit/bin:$PATH"

# Install library Python dengan Conda dan Pip
RUN conda install -n ml_bangkit 'numpy<2' pytorch torchvision torchaudio cpuonly -c pytorch -y && \
    conda install -n ml_bangkit notebook pandas scikit-learn matplotlib seaborn openpyxl lxml statsmodels opencv ipywidgets fastapi flask uvicorn shapely -y && \
    pip install imutils wordninja 'python-doctr[torch]==0.6' 'rapidfuzz==2.15.1' && \
    pip install 'tensorflow==2.10.1' 'tensorflow-datasets==4.8.0' 'tensorflow-hub==0.16.1' && \
    pip install 'tensorflow-recommenders==0.7.3' 'tensorflow-addons==0.18.0' 'tensorflow-probability==0.18.0' && \
    pip install 'PyYAML==5.3.1' 'tf-models-official==2.10.1'

# Expose port untuk aplikasi Flask
EXPOSE 8080

# Jalankan aplikasi Flask
CMD ["python", "main.py"]
