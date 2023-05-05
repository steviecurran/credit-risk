# credit-risk
Credit Risk Modelling For Dummies: But With Fewer Dummies

Standard credit risk analysis utilises scorecards which are built using only datasets with
categorical variables. This requires the continuous numerical features to be fine-classed (grouped
into discrete sets) and converted to dummy variables. Although the point of the scorecard is to
present the model in a simple way, this practice requires much convoluted pre-processing of the
data, which greatly bloats the size of the dataset and makes it more susceptible to containing
errors. Most importantly though, I find that, by retaining the numerical features, the predictive
power of the data is great improved, with a Gini coefficient of 0.98 (cf. 0.40 with fine-classing)
and Kolmogorov-Smirnov statistic of KS = 0.97 (cf. 0.30).

Although all of the processing was done in the python scripts mentioned in the report, a simplified 
run- through is included as the Jupyter notebooks PP.ipynb and ML.ipynb
