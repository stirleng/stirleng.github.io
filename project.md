## My Project

I applied machine learning techniques to investigate the associations between weather conditions and wildfires in Riverside County, CA. Below is my report.

***

## Introduction 

If you are one of the 39 million people currently living in California, then you have heard about our fire season. Wildfires are one of the greatest natural threats to the people of California. Every few years, preventative power outages cover the state, thousands are evacuated, and communities are burned to the ground.

![](assets/IMG/2023_stats.png){: width="800" }

When fires spread quickly, as is often the case with high winds, minimizing response time is critical in order to prevent polynomial growth in the area burned. If we can more effectively allocate resources towards high-fire-risk areas, then we can limit the resulting spread. Thus, being able to assess the probability of a fire is useful for fire safety. My goal with this project was to make such a tool, specifically, to predict the probability of a fire starting on a given day in Riverside County, California.

Fortunately, there are public datasets containing historical, quantitative weather and wildfire data. This means that I can use machine learning to attempt to solve this problem. Using the sklearn library, we can associate input variables (weather data from a given day) with an output variable (whether or not there was a fire in the area that day), train a model to detect how the input variables influence the output variable, and make a prediction as to the probability of the output variable taking on a certain value. This is the idea of supervised learning.

## Data

The dataset containing information about wildfires, obtained from [`fire.ca.gov`](https://fire.ca.gov), provides wide-ranging resources relating to every documented wildfire in California since 2013. However, we only care about a small subsection: the dates that fires started in Riverside County. To get this subset of the CSV file, I imported it as a pandas Dataframe, and filtered rows by checking the  'incident_county' column for entries of 'Riverside'. Then, I extracted the 'incident_date_created' field (don't worry, it's not an inside job) from every such row.

Preprocessing of the weather dataset, found on [`https://noaa.gov`](https://noaa.gov), was significantly more involved. As is often the case in the real world, the data collection tools were imperfect, and frequently failed to record any value. Further, the numerical data was still in string form, often (but not always) with units attached. Given the sheer size of the Dataframe, with ten years of hourly records, you can imagine why manually fixing these problems was infeasible. Instead, I wrote expressions that trimmed unnecessary fields, extracted the meaningful parts of others, and removed or replaced data where it was sparse.

![](assets/IMG/weather_preprocessing.PNG){: width="500" }

*Figure 1: The code described above.*

![](assets/IMG/dataframe_after_processing.PNG){: width="500" }

*Figure 2: A subsection of the processed weather data, displayed in a pandas Dataframe.*

## Modeling

As the goal of this project was to produce an estimation of the probability of a fire starting on a given day, I had to choose a model that would output probabilities. Of the models that we learned about in this class, only two can achieve this: linear regression and neural networks. As my dataset is significantly smaller once preprocessing is done (~3000), and neural networks tend to work better on immense datasets, I chose to use a logistic regression model.

First, I split the data into train and test sets.
```python
X = weather_df
X_train, X_test, y_train, y_test = train_test_split(X.drop(['is_fire'], axis=1), X['is_fire'], test_size=0.2)
```

Then, I used the Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset, as with an imbalanced dataset, the model could be very accurate just by predicting that there would be no fire, every single day.
```python
SMOTE = SMOTE()
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)
```

Then, I scaled the data features, to allow the model to more rapidly converge in training. I fit an sklearn StandardScaler() to the input features training data, and then transformed both the input features training data and the input features test data.
```python
#use a scitkit learn scaler so that we can use it later for new data points:
scaler_input = StandardScaler()
scaler_input.fit(X_train_SMOTE)
data_scaled_train = scaler_input.transform(X_train_SMOTE)
data_scaled_test = scaler_input.transform(X_test)

#put the scaled data back into a dataframe
data_scaled_train = pd.DataFrame(data_scaled_train)
data_scaled_train.index = X_train_SMOTE.index
mapcols = {}
for idx in range(len(X_train_SMOTE.columns)):
    mapcols[idx] = X_train_SMOTE.columns[idx]
X_train_SMOTE = data_scaled_train.rename(columns=mapcols)

data_scaled_test = pd.DataFrame(data_scaled_test)
data_scaled_test.index = X_test.index
mapcols = {}
for idx in range(len(X_test.columns)):
    mapcols[idx] = X_test.columns[idx]
X_test = data_scaled_test.rename(columns=mapcols)
```

Finally, I fit the model to the training data, and made predictions. Further below are the results.
```python
model = LogisticRegression()
# Train the model on the training data
model.fit(X_train_SMOTE, y_train_SMOTE)

# Make predictions on the test data
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```


## Results

![](assets/IMG/conf_mat.PNG){: width="500" }

*Figure 3: The confusion matrix, where 1 indicates a fire and 0 no fire on a given day.*

![](assets/IMG/ROC_curve.PNG){: width="500" }

*Figure 4: The ROC curve, with an AUC score of 0.80*

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

Limitations:
* Biased towards predicting fires because of resampling

## References

[back](./)

