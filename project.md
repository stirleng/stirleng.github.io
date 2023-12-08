## My Project

I applied machine learning techniques to investigate the associations between weather conditions and wildfires in Riverside County, CA. Below is my report.

***

## Introduction 

If you are one of the 39 million people currently living in California, then you have heard about our fire season. Wildfires are one of the greatest natural threats to the people of California. Every few years, preventative power outages cover the state, thousands are evacuated, and communities are burned to the ground.

![](assets/IMG/2023_stats.png){: width="800" }

When fires spread quickly, as is often the case with high winds, minimizing response time is critical in order to prevent polynomial growth in the area burned. If we can more effectively allocate resources towards high-fire-risk areas, then we can limit the resulting spread. Thus, being able to assess the probability of a fire is useful for fire safety. My goal with this project was to make such a tool.

Fortunately, there are public datasets containing historical, quantitative weather and wildfire data. This means that I can use machine learning to attempt to solve this problem. Using the sklearn library, we can associate input variables (weather data from a given day) with an output variable (whether or not there was a fire in the area that day), train a model to detect how the input variables influence the output variable, and make a prediction as to the probability of the output variable taking on a certain value. This is the idea of supervised learning.

## Data

The dataset containing information about wildfires, obtained from [`fire.ca.gov`](https://fire.ca.gov), provides wide-ranging resources relating to every documented wildfire in California since 2013. However, we only care about a small subsection: the dates that fires started in Riverside County. To get this subset of the CSV file, I imported it as a pandas Dataframe, and filtered rows by checking the  'incident_county' column for entries of 'Riverside'. Then, I extracted the 'incident_date_created' field (don't worry, it's not an inside job) from every such row.

Preprocessing of the weather dataset, found on [`https://noaa.gov`](https://noaa.gov), was significantly more involved. As is often the case in the real world, the data collection tools were imperfect, and frequently failed to record any value. Further, the numerical data was still in string form, often (but not always) with units attached. Given the sheer size of the Dataframe, with ten years of hourly records, you can imagine why manually fixing these problems was infeasible. Instead, I wrote expressions that trimmed unnecessary fields, extracted the meaningful parts of others, and removed or replaced data where it was sparse.

![](assets/IMG/weather_preprocessing.PNG){: width="500" }

*Figure 1: The code described above.*

![](assets/IMG/dataframe_after_processing.PNG){: width="500" }

*Figure 2: A subsection of the processed weather data, displayed in a pandas Dataframe.*

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

[back](./)

