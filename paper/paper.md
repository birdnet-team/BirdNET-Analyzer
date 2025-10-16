---
title: 'BirdNET-Analyzer: A bioacoustics analysis software'
tags:
  - Python
  - bioacoustics
  - machine learning
  - deep learning
  - sound identification
  - ecology
  - ornithology
  - audio analysis
  - bird sound recognition
  - birdnet
  - bird
authors:
  - name: Stefan Kahl
    equal-contrib: true
    affiliation: "1, 2"
    orcid: 0000-0002-2411-8877
  - name: Josef Haupt
    equal-contrib: true
    affiliation: 1
    orcid: 0009-0000-4646-6357
  - name: Max Mauermann
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Chemnitz University of Technology, Germany
   index: 1
   ror: 00a208s56
 - name: K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, Cornell University, USA
   index: 2
   ror: 00k86w020
date: 31 March 2025
bibliography: paper.bib
---

# Summary

BirdNET-Analyzer is an open-source toolkit for detection and classification of bird vocalizations in audio recordings using the well established BirdNET model [@kahl2021birdnet]. The application provides both a graphical user interface (GUI) and a command-line interface (CLI), which makes it easily accessible for users with varying levels of technical expertise.

The package supports macOS and Windows through dedicated installers, and it is also available as a Python package via PyPI to allow integration into custom pipelines.

# Statement of need

Monitoring avian biodiversity is an important tool for ecological research, conservation planning, and assessing environmental change, since birds are very sensitive to changing ecological factors. Manual listening and annotation are time-consuming, can be subjective, and impractical for the growing volume of acoustic data collected by autonomous recording units and community projects, such as BirdNET-Pi. As a result, there is a growing need for reliable, scalable, and user-friendly software to automate bird sound identification.

# Software description

The BirdNET-Analyzer is implemented in Python and built around the BirdNET deep learning model. The audio signals are converted into its spectrogram representation, processed by the model and predictions, with confidence scores and timestamps, are saved in the specified output format. Since Raven Pro is another popular tool for ecologists, the BirdNET-Analyzer supports the output of Raven compatible tables, but Audacity, csv and Kaleidoscope are also supported and can be combined at will.

The software can be considered as a collection of tools, that represent the quasi workflow of ecological work. It consists of the audio analysis which outputs the scores and timestamps, the extraction of the audio segments with that include some target species and validation of the extracted segments. In addition the BirdNET-Analyzer also enables users to train a custom classification layer if, for example the target species is not included in the standard BirdNET model or when a specialized classifier is necessary. The custom classification layer will either replace the existing one of the default model or can be appended to the existing one, also leaving the predictions from BirdNET in the output. These newly trained models can also be evaluated directly in the application using the "Evaluation" module, which allows users to compare predictions with annotations and outputs metrics to evaluate the trained model.

![The graphical interface of the BirdNET-Analyzer](fig/gui.png)

Embeddings can be used to compare calls and are essentially a compressed version of the audio signal. The application allows users to extract the embeddings and save them in a database using the perch-hoplite. These databases can be used to search for similar embeddings usind an audio sample.
