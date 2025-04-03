# Build from Python slim
FROM python:3.11

# By default installing all dependencies, but this can be changed with build-arg
ARG TARGET='all'

# Install required packages while keeping the image small
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Import all scripts
COPY . ./

# Install required Python packages
RUN pip3 install --no-cache-dir .[${TARGET}]

WORKDIR /input

# Add entry point to run the script
ENTRYPOINT [ "python3" ]