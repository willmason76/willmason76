# Crime Prediction in the United States

The combination of compute commoditization, open source algorithms, and the availability of city specific crime data has made the reality of spatio-temporal crime prediction a reality for law enforcement. This file is an overview of a research project meant to explore the limits of accuracy for predicting future violent crimes. 

Four cities were processed, evaluated and visualized, and the results were in line with the leading publicly available research papers. Here we present our methods and results.

## Methods

Our data is organized in shape (space, time, features), and we use a traditional Recurrent Neural Network, in this case a ConvLSTM2D model. While new architectures such as transformers might in theory allow for temporal tendencies to be better learned over longer context windows, that decision comes with the drawback of having to collapse dimensions to fit within that structure, losing some of the learnable memory (fix this).

![Crime](https://github.com/willmason76/willmason76/blob/main/algo.png)

## Results

For evaluation, we use the custom Area Under the Curve metric described above, where we treat a correct prediction if a ground truth y occurred within one timestep backward or forward from yhat.

AUC for Las Vegas
![Crime](https://github.com/willmason76/willmason76/blob/main/ROC.png)
<img src="([https://github.com/willmason76/willmason76/blob/main/ROC.png](https://raw.githubusercontent.com/willmason76/willmason76/main/ROC.png
)" alt="Crime" width="200" height="100">


AUC for New Orleans
![Crime](https://github.com/willmason76/willmason76/blob/main/seven_and_two.png)

## Visualizations

