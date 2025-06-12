Running in Docker    
=================

BirdNET-Analyzer supports running CLI tools within a Docker containerized environment. To get started, follow these steps:

**Step 1: Clone the Repository**

Begin by cloning the official BirdNET-Analyzer repository:

.. code:: bash
    
    git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
    cd BirdNET-Analyzer


**Step 2: Install Docker**

Ensure that Docker is installed and the Docker daemon is running on your machine. For more information on that, please consult the official `Docker documentation <https://docs.docker.com>`_.


**Step 3: Build the Docker Image**

You need to build the Docker image using the provided `Dockerfile`. The build can be customized based on the needs of your project. Multiple build targets are supported, and for more details, check the `pyproject.toml` file.

By default, Docker will build an image containing all the necessary dependencies for all targets. If you require a specific target, set the build argument `TARGET` to the desired value. For example, if you want to build Docker container with dependencies for the training component, use the following command:

.. code:: bash

    docker build -t birdnet --build-arg TARGET=train .


**Step 4: Run the Docker Image**

Once the image is built, it will appear in your list of Docker images. To run the container, use the following command (assuming your image is named `birdnet`):

.. code:: bash

    docker run birdnet:latest

.. note::

    To ensure the software functions correctly, you need to attach two directories: one for input files and one for output files. The Docker image will use the `/input` directory for input files (which serves as the working directory inside the container) and the `/output` directory for output files. Use volume mapping to map these directories to your local directories.

    .. code:: bash

        #Assuming I have ~/my_input_folder with owls.wav file and I want to output results to ~/my_output_folder
        
        docker run \
        -v ~/my_input_folder:/input \ # attaching input directory
        -v ~/my_output_folder:/output \ # attaching output directory
        birdnet:latest -m birdnet_analyzer.analyze -o /output owls.wav # refering the file in the input directory

Docker container’s entry point is `python3`. You need to add `-m birdnet_analyzer.{module-name}` with the name of the module to start. You can pass additional arguments directly via the Docker `run` command interface as needed.