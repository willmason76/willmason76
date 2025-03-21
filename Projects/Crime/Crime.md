# Vegas Labs - Crime Prediction in the United States 


The combination of compute commoditization, open source algorithms, and the availability of city specific crime data has made the reality of spatio-temporal crime prediction a reality for law enforcement. This file is an overview of a research project meant to explore the limits of accuracy for predicting future violent crimes. 

Four cities were processed, evaluated and visualized, and the results were in line with the leading publicly available research papers. Here we present our methods and results.

## Data

For each city, we use a representative dataset of reported crimes. This is in contrast to actual crimes that have been adjudicated through criminal court. To our knowledge, that dataset is not available, but theoretically you are balancing imperfections and bias in the criminal justice system versus those of citizens reporting criminal activity. We believe citizens are the better barometer, whether we have a choice in datasets or not. 

We set our target for violent crimes against the [FBI's Uniform Crime Report](https://ucr.fbi.gov/additional-ucr-publications/ucr_handbook.pdf) value. While some cities do better than others in labeling and segregating these values, our goal is to train and inference solely on UCR values < 10.

## Methods

Our data is organized in shape (space, time, features), and we use a traditional Recurrent Neural Network, in this case a ConvLSTM2D model. While new architectures such as transformers might in theory allow for temporal tendencies to be better learned over longer context windows, that decision comes with the drawback of having to collapse dimensions to fit within that structure, losing some of the learnable memory (fix this).

![Crime](https://github.com/willmason76/willmason76/blob/main/Projects/Crime/algo.png)

## Results

For evaluation, we use the custom Area Under the Curve (AUC) score described above, where we treat a correct prediction if a ground truth y occurred within one timestep backward or forward from ŷ.

Roc Curve & AUC for Four Cities
<div style="display: flex; align-items: center;">
    <img src="https://raw.githubusercontent.com/willmason76/willmason76/main/Projects/Crime/ROC_combined.png" alt="Crime 1" width="500" height="400">
</div>


