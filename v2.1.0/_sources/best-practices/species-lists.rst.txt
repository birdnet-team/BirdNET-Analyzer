Creating Your Own Species List
==============================

When editing your own `species_list.txt` file, make sure to copy species names from the labels file of each model.

You can find label files in the checkpoints folder, e.g., `checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt`.

Species names need to consist of `scientific name_common name` to be valid.

You can generate a species list for a given location using :ref:`species.py <cli-species>`.

Practical Information and Considerations
----------------------------------------

**Understanding the GeoModel**

The BirdNET Species Range Model V2.4 - V2 uses eBird checklist frequency data to estimate the range of bird species and the probability of their occurrence given latitude, longitude, and week of the year. eBird relies on citizen scientists to collect bird species observations around the world. Due to biases in these data, some regions such as North and South America, Europe, India, and Australia are well represented in the data, while large parts of Africa or Asia are underrepresented.

In cases where eBird does not have enough observations (i.e., checklists), the data "only" contain binary filter data of likely species that could occur in a given location. Therefore, the training data for our biodiversity model is a mixture of actual observations and filter data curated by experts. We included all locations for which at least 10 checklists are available for each week of the year, and randomly added other locations with a 3% probability.

**Limitations of the GeoModel**

- **Data Coverage**: The model works well in regions with good eBird data coverage, such as North and South America, Europe, India, and Australia. In other regions, the lack of eBird observations means the resulting species lists may not reflect actual probabilities of occurrence.
- **Binary Filter Data**: In areas with insufficient eBird data, the model relies on binary filter data, which may not be as accurate as actual observations.
- **Seasonal Variations**: The model accounts for seasonal variations in bird presence, but the accuracy depends on the availability of data for each week of the year.

**Creating Custom Species Lists**

If you know which species to expect in your area, it is recommended to compile your own species list. This can help improve the accuracy of BirdNET-Analyzer for your specific use case.

1. **Collect Species Names**: Use the labels file from the model checkpoints to get the correct species names. Ensure the names are in the format `scientific name_common name`.
2. **Generate Species List**: Use the `species.py` script to generate a species list for a given location and time. This script uses the GeoModel to predict species occurrence based on latitude, longitude, and week of the year.

**Example of Training Data**

Here is an example of what the training data for a given location (Chemnitz) looks like:

.. code:: python

    'gretit1': [72, 90, 98, 93, 96, 88, 95, 94, 99, 99, 93, 92, 90, 96, 85, 97, 89, 78, 67, 68, 48, 39, 35, 40, 49, 49, 49, 51, 48, 55, 55, 73, 60, 64, 62, 63, 72, 72, 72, 67, 66, 80, 63, 74, 67, 76, 88, 70], 
    'carcro1': [62, 81, 83, 82, 85, 75, 90, 75, 83, 80, 76, 80, 84, 90, 72, 73, 83, 67, 70, 75, 54, 48, 42, 55, 51, 53, 55, 49, 55, 53, 55, 62, 57, 55, 66, 69, 63, 65, 69, 63, 59, 74, 61, 63, 76, 79, 69, 60], 
    'eurbla': [55, 80, 84, 92, 71, 70, 72, 84, 85, 86, 82, 95, 88, 92, 86, 91, 90, 75, 87, 81, 84, 72, 69, 62, 67, 70, 57, 66, 55, 56, 49, 32, 36, 37, 41, 49, 55, 62, 57, 58, 41, 37, 58, 67, 69, 64, 69, 49], 
    'blutit': [67, 83, 92, 93, 96, 83, 87, 93, 96, 90, 82, 80, 84, 88, 58, 79, 74, 52, 46, 36, 34, 29, 25, 26, 39, 43, 36, 43, 47, 42, 49, 48, 49, 51, 45, 52, 61, 64, 55, 55, 65, 72, 62, 71, 66, 67, 69, 64], 
    'grswoo': [61, 84, 80, 80, 90, 83, 85, 77, 76, 82, 72, 77, 77, 78, 64, 76, 81, 69, 73, 75, 66, 44, 46, 41, 47, 41, 38, 44, 42, 42, 52, 68, 37, 35, 38, 43, 44, 41, 43, 41, 49, 61, 41, 49, 48, 47, 67, 47], 
    'cowpig1': [9, 10, 3, 3, 16, 16, 30, 54, 65, 61, 69, 76, 83, 81, 80, 86, 80, 71, 68, 78, 68, 69, 79, 68, 76, 69, 69, 79, 70, 70, 68, 73, 64, 63, 58, 54, 53, 49, 53, 56, 44, 21, 33, 38, 45, 43, 5, 11],
    'eurnut2': [43, 76, 88, 82, 79, 78, 91, 84, 92, 86, 76, 77, 75, 85, 69, 75, 60, 34, 47, 58, 34, 24, 33, 33, 31, 23, 28, 25, 23, 21, 23, 52, 26, 26, 31, 28, 25, 29, 32, 23, 47, 46, 24, 31, 30, 36, 61, 53], 
    'comcha': [26, 33, 30, 33, 34, 34, 39, 48, 70, 75, 80, 83, 80, 90, 76, 85, 80, 74, 77, 74, 59, 52, 51, 40, 34, 44, 33, 31, 22, 15, 17, 21, 17, 18, 26, 34, 44, 48, 53, 49, 31, 27, 33, 39, 44, 39, 30, 28]

**Example of Model Predictions**

If we query the trained model for the same location as above, we get these values for great tits:

.. code:: python

    'gretit': [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 98, 98, 98, 98, 98, 97, 97, 97, 97, 97, 97, 98, 98, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]

**Conclusion**

Overall, the model works well in regions with good data coverage. In other regions, the lack of eBird observations means the resulting species lists may not reflect actual probabilities of occurrence. Nevertheless, these lists can be used to filter for species that may or may not occur in these locations.

By understanding the limitations and capabilities of the GeoModel, you can make informed decisions when creating and using custom species lists for BirdNET-Analyzer.

See this post in the discussion forum for more details: `Species range model details <https://github.com/birdnet-team/BirdNET-Analyzer/discussions/234>`_