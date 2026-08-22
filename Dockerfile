# Local viewer/runner for the notebooks. The actual GPU training happens in Colab
# (see the notebooks' own setup cells) — this image exists so anyone cloning the
# repo can open CitySegmentation.ipynb / HRSID_Segmentation.ipynb in a reproducible
# environment without hunting down dependency versions themselves.
FROM python:3.10-slim

WORKDIR /workspace

# opencv needs these at the OS level even with the opencv-python-headless wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir jupyterlab

COPY . .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
