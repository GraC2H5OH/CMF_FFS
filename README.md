# CMF_FFS
## Contents:
- [Backstory](#backstory)
  - [Problems](#problems)
  - [Solution](#solution)
- [Goals](#goals)
- [What will be done](#what-will-be-done)
- Models results
- Used methods and models description
- Soon
- [Sources](#sources)

### Backstory
#### Problems
Modeling of the company's operations is mainly done by people and they can make mistakes.
![problem](https://github.com/GraC2H5OH/CMF_FFS/blob/main/pics/problems.png)

However experienced economists and analytics could do something different instead of this. Creating a new model takes weeks. 
#### Solution
Let's use machine learning to reduce the error and eliminate the human factor.

### Goals
Soon
### What will be done
- Basic EDA
- Metrics selection
- Fitting baseline model
- Advanced models
- Cross-validation
- feature selection/engineering
- Interpretation of models results(if possible)

### Models results
| Models and methods                                          | MAPE   | WAPE  | MSE   |R<sup>2</sup><sub>adj</sub>|
|-------------------------------------------------------------|--------|-------|-------|----------------|
| Simple linear regressions<br>for each company               | 0.59   |0.57   |702373 |11.57           |
|Simple linear regression<br>for each companies               |26003508|0.3e+06|4.6e+20|-479824272578422|
|Relaxed Lasso with <br> top 20 features for <br> each company|0.355   |0.35   |5390113|7.8             |
|Relaxed Lasso with <br> top 10 features for <br> each company|0.197   |0.19   |68719  |2.3             |


### Sources
i will use [this](https://www.kaggle.com/datasets/jarbol/oil-gas-predict) datasets
