## My Project

I applied machine learning techniques to investigate the associations between weather conditions and wildfires in Riverside County, CA. Below is my report.

***

## Introduction 

If you are one of the 39 million people currently living in California, then you have heard about our fire season. Wildfires are one of the greatest natural threats to the people of California. Every few years, preventative power outages cover the state, thousands are evacuated, and communities are burned to the ground.

![](assets/IMG/2023_stats.png){: width="800" }

When fires spread quickly, as is often the case with high winds, minimizing response time is critical in order to prevent polynomial growth in the area burned. If we can more effectively allocate resources towards high-fire-risk areas, then we can limit the resulting spread. Thus, being able to assess the probability of a fire is useful for fire safety. My goal with this project was to make such a tool.

Fortunately, there are public datasets containing historical, quantitative weather and wildfire data. This means that I can use machine learning to attempt to solve this problem. Using the sklearn library, we can associate input variables (weather data from a given day) with an output variable (whether or not there was a fire in the area that day), train a model to detect how the input variables influence the output variable, and make a prediction as to the probability of the output variable taking on a certain value. This is the idea of supervised learning.

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modeling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] noaa.gov
[2] fire.ca.gov

[back](./)

