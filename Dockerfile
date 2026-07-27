# Match the Python version used to build the release (see publish.yml / CI matrix)
FROM python:3.12-slim

# Install required packages while keeping the image small
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg  && rm -rf /var/lib/apt/lists/*

# Import all scripts
WORKDIR /app
COPY . ./

# Install the package. On x86_64, replace the default TensorFlow build with the
# CPU-only wheel: BirdNET-Analyzer runs inference on CPU (the birdnet library
# uses ai-edge-litert), so the GPU kernels bundled in the default x86_64 wheel
# (~360 MB) are dead weight. tensorflow-cpu is only published for x86_64
# (no linux/arm64 wheel exists), and the arm64 tensorflow wheel ships no GPU
# kernels anyway, so on arm64 we keep the default build. TARGETARCH is provided
# automatically by BuildKit/buildx.
ARG TARGETARCH
RUN pip3 install --no-cache-dir . \
    && if [ "$TARGETARCH" = "amd64" ]; then \
         pip3 uninstall -y tensorflow \
         && pip3 install --no-cache-dir "tensorflow-cpu>=2.20"; \
    fi

# Add entry point to run the script
ENTRYPOINT [ "python3" ]
CMD [ "-m", "birdnet_analyzer.analyze" ]
