# Citation Sentiment Reflects Multiscale Sociocultural Norms
Data used in the manuscript are available upon request from the corresponding author at dsb@seas.upenn.edu.

Which script does what:</br>
- ```generate_data.ipynb``` pre-processes; prepare data for figure making; this script is the main script one needs
- ```chatgpt_validation_rate_300.ipynb``` sends training and validation sentences to trained model to rate sentiment for inter-annotator agreement
- ```chatgpt_validation_process_output.ipynb``` processes inter-annotator agreement output from ```chatgpt_validation_rate_300.ipynb``` and make supplement figure
- ```plot_.ipynb``` all ```.ipynb``` files with a ```plot_``` prefix are figure making scripts; run only after finishing ```generate_data.ipynb``` entirely
- The rest are all helper scripts
