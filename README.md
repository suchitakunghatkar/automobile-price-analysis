# Automobile Price Analysis & Prediction

## 📌 Project Overview

This project analyzes an automobile dataset to identify the key factors associated with vehicle prices and build machine-learning models for automobile price prediction.

The project combines **Python, Pandas, NumPy, Exploratory Data Analysis (EDA), statistical analysis, data visualization, and machine learning** to understand pricing patterns and evaluate predictive performance.

---

## 🎯 Business Objective

The objective of this project is to:

* Understand the characteristics of automobiles in the dataset.
* Identify factors strongly associated with automobile prices.
* Analyze relationships between vehicle specifications and pricing.
* Detect missing values, outliers, skewness, and multicollinearity.
* Build regression models to predict automobile prices.
* Compare Linear Regression and Random Forest performance.
* Identify the most important features influencing model predictions.

---

## 📊 Dataset

The dataset contains **205 automobile records and 26 original features** covering:

* Vehicle manufacturer
* Fuel type
* Body style
* Drive-wheel configuration
* Engine characteristics
* Vehicle dimensions
* Horsepower
* Mileage
* Automobile price

After data cleaning, the final modeling dataset contained:

* **201 records**
* **26 columns**
* **0 missing values**
* **0 duplicate records**

The four records with missing target values (`price`) were removed because the target variable cannot be reliably imputed for supervised price prediction.

---

# 🔧 Technologies & Tools

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Statsmodels**
* **Jupyter Notebook**

### Machine Learning

* Linear Regression
* Random Forest Regression
* One-Hot Encoding
* Train-Test Split
* Model Evaluation

### Statistical Analysis

* Pearson Correlation
* P-value / Statistical Significance
* IQR Outlier Detection
* Skewness Analysis
* Variance Inflation Factor (VIF)

---

# 🧹 1. Data Cleaning

The initial dataset contained missing values in several columns.

### Cleaning techniques applied:

* Identified missing values and their percentages.
* Used **median imputation** for numerical variables.
* Used **mode imputation** for the categorical `num-doors` variable.
* Converted numerical fields to appropriate data types.
* Converted `num-doors` from text values to numerical values.
* Converted `num-cylinders` from text values to numerical values.
* Removed records with missing `price`.
* Checked for duplicate records.
* Validated the final dataset.

### Final Data Quality

| Check           | Result |
| --------------- | -----: |
| Initial Records |    205 |
| Final Records   |    201 |
| Features        |     26 |
| Missing Values  |      0 |
| Duplicate Rows  |      0 |

---

# 📈 2. Exploratory Data Analysis

## 2.1 Categorical Analysis

Some major observations:

* **Toyota** was the most represented manufacturer with **32 vehicles**.
* **90.0%** of vehicles used gasoline and **10.0%** used diesel.
* **Sedan** was the most common body style at **46.8%**.
* **FWD** was the dominant drive-wheel configuration at **58.7%**.
* **Standard aspiration** represented **82.1%** of vehicles.
* **OHC** was the most common engine type at **72.1%**.
* **Front-engine vehicles** represented **98.5%** of the dataset.
* **MPFI** was the most common fuel system at **45.8%**.

---

# 📊 3. Numerical Analysis

| Variable    | Minimum | Median |     Mean | Maximum |
| ----------- | ------: | -----: | -------: | ------: |
| Price       |   5,118 | 10,295 |   13,207 |  45,400 |
| Horsepower  |      48 |     95 |   103.31 |     262 |
| Engine Size |      61 |    120 |   126.88 |     326 |
| Curb Weight |   1,488 |  2,414 | 2,555.67 |   4,066 |
| City MPG    |      13 |     24 |    25.18 |      49 |
| Highway MPG |      16 |     30 |    30.69 |      54 |

The price variable was positively skewed because the mean price (**13,207**) was higher than the median price (**10,295**).

---

# 🔎 4. Price Relationship Analysis

Pearson correlation analysis identified several strong relationships with automobile price.

| Feature             | Correlation with Price |
| ------------------- | ---------------------: |
| Engine Size         |             **0.8723** |
| Curb Weight         |             **0.8344** |
| Horsepower          |             **0.8105** |
| Highway MPG         |            **-0.7047** |
| City MPG            |            **-0.6866** |
| Width               |             **0.7513** |
| Number of Cylinders |             **0.7086** |
| Length              |             **0.6906** |

### Key Insight

Engine size had the strongest positive correlation with price, followed by curb weight and horsepower.

Highway MPG showed a strong negative relationship with price.

These relationships represent **associations within this dataset and should not be interpreted as causal relationships**.

---

# 📦 5. Price vs Categorical Variables

### Manufacturer

* Jaguar had the highest average price: **34,600**
* Mercedes-Benz: **33,647**
* Porsche: **31,400.50**
* Chevrolet had the lowest average price: **6,007**

Some manufacturers had very small sample sizes, so these averages should be interpreted cautiously.

### Body Style

* Hardtop: **22,208.50** average price
* Convertible: **21,890.50**
* Sedan: **14,459.76**
* Wagon: **12,371.96**
* Hatchback: **9,957.44**

### Fuel Type

* Diesel: **15,838.15**
* Gas: **12,916.41**

### Drive Wheels

* RWD: **19,757.61**
* 4WD: **10,241.00**
* FWD: **9,244.78**

### Aspiration

* Turbo: **16,254.81**
* Standard: **12,542.18**

These are dataset-specific group comparisons and do not establish causation.

---

# 📐 6. Advanced Statistical Analysis

## 6.1 Pearson Correlation Significance

Statistical testing showed significant relationships between price and:

* Engine size: **r = 0.8723**
* Curb weight: **r = 0.8344**
* Horsepower: **r = 0.8105**
* Highway MPG: **r = -0.7047**

All four relationships had **p < 0.05**.

---

## 6.2 Outlier Detection

The IQR method was used to identify potential outliers.

| Variable    | Potential Outliers |
| ----------- | -----------------: |
| Price       |                 14 |
| Engine Size |                 10 |
| Horsepower  |                  5 |
| Curb Weight |                  2 |

The potential outliers were investigated conceptually rather than automatically removed. Many represented legitimate high-performance or premium vehicles.

---

## 6.3 Price Distribution & Transformation

Original price skewness:

**1.8097**

After applying `log1p()`:

**0.6787**

The log transformation substantially reduced the right skewness of the price distribution.

---

# ⚠️ 7. Multicollinearity Analysis

Variance Inflation Factor (VIF) was used to identify multicollinearity among numerical predictors.

| Feature     |       VIF |
| ----------- | --------: |
| City MPG    | **23.88** |
| Highway MPG | **23.57** |
| Curb Weight | **12.62** |
| Engine Size | **11.59** |
| Horsepower  |  **6.37** |
| Length      |  **6.21** |

City MPG and Highway MPG were particularly highly correlated:

**Correlation = 0.9720**

This indicates substantial overlap between these predictors.

Multicollinearity was considered during model interpretation rather than automatically removing all high-VIF variables.

---

# 🤖 8. Machine Learning

## 8.1 Feature Preparation

The target variable was:

```text
price
```

The `log_price` column was excluded from the predictors to avoid target leakage.

Categorical variables were encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

Numerical variables were passed through the preprocessing pipeline.

The dataset was split into:

* **80% training data**
* **20% testing data**

`random_state=42` was used for reproducibility.

---

# 📉 9. Model Performance

Two regression models were evaluated.

| Model                 |          MAE |         RMSE |         R² |
| --------------------- | -----------: | -----------: | ---------: |
| **Linear Regression** | **1,721.20** | **2,802.64** | **0.9358** |
| Random Forest         |     1,855.90 |     2,885.01 |     0.9320 |

### Interpretation

Linear Regression achieved:

* **MAE = 1,721.20**
* **RMSE = 2,802.64**
* **R² = 0.9358**

The R² value indicates that the model explained approximately **93.58% of the variation in automobile prices on the test set**.

For this particular train-test split, Linear Regression performed slightly better than Random Forest across all three evaluation metrics.

---

# 🔍 10. Feature Importance

## Random Forest

The most important features were:

| Rank | Feature             | Importance |
| ---: | ------------------- | ---------: |
|    1 | **Curb Weight**     | **42.88%** |
|    2 | **Engine Size**     | **28.85%** |
|    3 | Horsepower          |      6.22% |
|    4 | Highway MPG         |      5.71% |
|    5 | City MPG            |      3.89% |
|    6 | Width               |      2.66% |
|    7 | Wheel Base          |      1.58% |
|    8 | Number of Cylinders |      1.21% |
|    9 | Length              |      0.87% |
|   10 | BMW                 |      0.87% |

Curb weight and engine size together represented approximately **71.73%** of the Random Forest feature-importance measure.

This is consistent with the earlier correlation analysis, where engine size and curb weight also showed strong relationships with price.

---

# 📌 11. Key Project Insights

1. **Engine size, curb weight, and horsepower** showed strong positive relationships with automobile price.
2. **Highway MPG and city MPG** showed negative relationships with price.
3. Automobile prices varied substantially across **manufacturers, body styles, fuel types, drive wheels, and aspiration types**.
4. The price distribution was strongly right-skewed, and `log1p()` reduced the skewness.
5. Several predictors showed substantial multicollinearity, particularly city MPG and highway MPG.
6. **Curb weight and engine size were the two most important Random Forest features.**
7. Linear Regression achieved an **R² of 0.9358** and performed slightly better than Random Forest on the selected test set.
8. The analysis demonstrates how EDA, statistical analysis, and machine learning can be combined to understand and predict automobile prices.

---

# 🧠 12. Skills Demonstrated

### Data Analysis

* Data cleaning
* Missing-value treatment
* Data type conversion
* Exploratory data analysis
* Grouped analysis
* Aggregation
* Statistical analysis

### Python

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Statistics

* Pearson correlation
* Statistical significance
* Skewness
* IQR outlier detection
* Multicollinearity
* VIF

### Machine Learning

* Feature preprocessing
* One-hot encoding
* Pipeline creation
* Train-test split
* Linear Regression
* Random Forest Regression
* MAE
* RMSE
* R²
* Feature importance

---

# 📁 13. Suggested Repository Structure

```text
automobile-price-analysis/
│
├── data/
│   └── cars_data.csv
│
├── notebooks/
│   └── automobile_price_analysis.ipynb
│
├── images/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── price_vs_engine_size.png
│   ├── price_vs_horsepower.png
│   └── feature_importance.png
│
├── README.md
└── requirements.txt
```

---

# 🚀 14. Conclusion

This project demonstrates an end-to-end approach to automobile price analysis, beginning with data cleaning and exploratory analysis and progressing through statistical analysis, feature preparation, machine-learning modeling, and model interpretation.

The analysis identified **engine size, curb weight, horsepower, and fuel efficiency** as important variables associated with automobile pricing. Among the evaluated models, Linear Regression achieved the strongest performance on the selected test set with an **R² of 0.9358**.

The project provided practical experience in transforming raw automobile data into analytical insights and predictive models using Python.
