# Match the Python version used to build the release (see publish.yml / CI matrix)
FROM python:3.13-slim

# uv installs the (large scientific) dependency tree far faster than pip via
# parallel downloads and a faster resolver. Pin the version for reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

# Install required packages while keeping the image small.
# git is only needed to resolve the temporary birdnet direct reference in
# pyproject.toml; drop it again once birdnet is installed from PyPI.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the third-party dependencies from the project metadata alone, before
# the source is copied. Editing a .py file then only invalidates the two cheap
# layers at the bottom instead of re-resolving and re-downloading ~2.5 GB of
# wheels.
#
# On x86_64, replace the default TensorFlow build with the CPU-only wheel:
# BirdNET-Analyzer runs inference on CPU (birdnet.load(..., "tf", ...) executes
# the TFLite model through tf.lite), so the GPU kernels bundled in the default
# x86_64 wheel (~360 MB) are dead weight. tensorflow-cpu is only published for x86_64
# (no linux/arm64 wheel exists), and the arm64 tensorflow wheel ships no GPU
# kernels anyway, so on arm64 we keep the default build. TARGETARCH is provided
# automatically by BuildKit/buildx.
COPY pyproject.toml ./
ARG TARGETARCH
RUN python3 -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && if [ "$TARGETARCH" = "amd64" ]; then \
         uv pip uninstall --system tensorflow \
         && uv pip install --system --no-cache "tensorflow-cpu>=2.20"; \
       fi

# Bake the models into the image. The birdnet dependency otherwise fetches them
# on first use, and under the documented `docker run --rm` usage every run is a
# first run: the download dominated the runtime of a short analysis and made the
# image unusable without network access. These are the same birdnet.load() calls
# the runtime makes (see run_inference/run_geomodel in birdnet_analyzer/model_utils.py),
# so the cache they populate under /root/.local/share/birdnet is exactly what
# the container - also root - looks for later. The geo model is only needed for
# --lat/--lon species filtering but is cheap relative to the image. "2.4"/en_us
# track the defaults in birdnet_analyzer/config.py and model_utils.py; if those
# move, the image simply falls back to downloading at runtime.
RUN python3 -c "import birdnet; birdnet.load('acoustic', '2.4', 'tf', lang='en_us'); birdnet.load('geo', '2.4', 'tf', lang='en_us')"

# Import all scripts
COPY . ./

# Dependencies are already installed above, so this only registers the package.
RUN uv pip install --system --no-cache --no-deps .

# Add entry point to run the script
ENTRYPOINT [ "python3" ]
CMD [ "-m", "birdnet_analyzer.analyze" ]
