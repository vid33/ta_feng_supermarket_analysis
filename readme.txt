
Analysis of the "Ta Feng" data set. 

NOTE: Many more details are provided in comments in individual files.

The order in which things should be run is:

1) preprocess.py: This script cleans up the original data from the "D11-02" folder (which should be present), and saves the resulting files into the "data" folder. It can be run for individual months, the entire period. 'all_periodic' option dumps the data from holiday periods (e.g. Chinese N.Y.), and also makes sure that the data is periodic in weeks - since this is the most obvious periodicity in user behavior.

2) visualize.py: Loads the cleaned up data. Spits out various plots, and some data analysis, in order to get a feel for the data.

3) engineer_features.py: This will create a features array for each supermarket user. So far I have only constructed a few features, and many more are needed for a more thorough analysis. Features are saved to files in the data folder.

4) analyse_cluster.py and analyse_supervised.py: These load the features data from the "data" folder constructed above, and preform some basic clustering analysis (at the moment just k-means), and softmax regression that attempts to predict the user age. I've used tensorflow, so it should be easy to generalise to a more sophisticated analysis.




