# Match the Python version used to build the release (see publish.yml / CI matrix)
FROM python:3.12-slim

# Install required packages while keeping the image small
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg  && rm -rf /var/lib/apt/lists/*

# Import all scripts
WORKDIR /app
COPY . ./

# Install the package, then replace the default TensorFlow build with the
# CPU-only wheel. BirdNET-Analyzer runs inference on CPU (the birdnet library
# uses ai-edge-litert), so the GPU-enabled TensorFlow build is dead weight.
RUN pip3 install --no-cache-dir . \
    && pip3 uninstall -y tensorflow \
    && pip3 install --no-cache-dir "tensorflow-cpu>=2.20"

# Add entry point to run the script
ENTRYPOINT [ "python3" ]
CMD [ "-m", "birdnet_analyzer.analyze" ]
