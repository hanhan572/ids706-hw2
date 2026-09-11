# Wine Quality Data Analysis and Rust Ownership

## Project Overview

This project is Part 1 of a three-week data analysis project. I used a merged red and white wine quality dataset to practice data analysis with Pandas, including data inspection, cleaning, filtering, grouping, visualization, and machine learning.

I also used Polars to perform the same data-processing workflow and compared its execution time with Pandas.

The project also includes a Rust Jupyter notebook exploring Rust syntax, mutability, ownership, cloning, borrowing, and memory management.

## Project Goal

The goal of this project is to explore the characteristics of red and white wines, identify patterns related to wine quality, and build a simple Linear Regression model to predict wine quality using physicochemical features.

## Dataset

The original dataset contains 6,497 observations and 13 variables. It includes both red and white wines.

The variables include:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality
- type

The `quality` variable represents the wine quality score, while `type` identifies whether the wine is red or white.

Source: [Red and White Wine Quality dataset on Kaggle](https://www.kaggle.com/datasets/amirmohamadrezaie/red-and-white-wine-quality)

## Data Inspection

I loaded the dataset using Pandas and inspected it using:

- `head()` to view the first five rows
- `info()` to examine column names, data types, and non-null values
- `describe()` to calculate summary statistics
- `isnull()` to check for missing values
- `duplicated()` to check for duplicate rows

The original dataset contains no missing values.

The original dataset contains 1,177 duplicate rows.

## Data Cleaning

I removed duplicate observations using `drop_duplicates()` and reset the DataFrame index.

After removing duplicates, the dataset contains:

- 5,320 observations
- 13 variables
- 0 duplicate rows

The cleaned dataset was used for the subsequent filtering, grouping, visualization, and machine learning analysis.

## Filtering

I filtered the cleaned dataset to identify wines with a quality score of 7 or higher.

There are 1,009 high-quality wines in the cleaned dataset.

## Grouping

I grouped the cleaned dataset by wine type and calculated the average quality score, average alcohol content, and number of observations.

The results were:

| Wine Type | Average Quality | Average Alcohol | Count |
|-----------|----------------:|----------------:|------:|
| Red | 5.623 | 10.432 | 1,359 |
| White | 5.855 | 10.589 | 3,961 |

White wines in this dataset have a slightly higher average quality score and slightly higher average alcohol content than red wines.

## Visualization

I created a scatter plot of alcohol content versus wine quality because a scatter plot makes it easy to examine the relationship between a numerical predictor (`alcohol`) and the target variable (`quality`).

The plot suggests a positive relationship between alcohol content and wine quality. Wines with higher alcohol content tend to receive higher quality scores, although there is substantial overlap across quality levels.

![Alcohol Content vs Wine Quality](wine_quality_scatter.png)

## Machine Learning

I experimented with a Linear Regression model to predict wine quality.

The model inputs were the 11 numerical physicochemical characteristics of the wines. The target variable was `quality`.

The categorical `type` variable was excluded from this initial model.

The cleaned dataset was divided into:

- 80% training data
- 20% testing data

A fixed `random_state=42` was used so that the train-test split is reproducible.

### Model Results

- Mean Absolute Error (MAE): 0.561
- R-squared: 0.302

The MAE indicates that the predicted wine quality differs from the actual quality score by approximately 0.56 points on average.

The R-squared value indicates that the Linear Regression model explains approximately 30.2% of the variation in wine quality.

This suggests that the physicochemical characteristics contain useful information for predicting wine quality, but the simple linear model does not explain all of the variation.

## Pandas vs Polars Performance Comparison

As an additional experiment, I used both Pandas and Polars to perform the same data-processing workflow:

1. Read the CSV file
2. Remove duplicate rows
3. Group the data by wine type
4. Calculate average quality
5. Calculate average alcohol content
6. Count the observations in each group

Both Pandas and Polars produced the same summary statistics.

The execution times from one run were:

- Pandas: 0.010602 seconds
- Polars: 0.098031 seconds

In this experiment, Pandas was faster than Polars. The original dataset contains only 6,497 rows, so the overhead associated with initializing and executing the Polars workflow may be relatively large compared with the amount of data-processing work.

This small benchmark does not imply that Pandas is always faster than Polars. Performance can depend on dataset size, operation type, hardware, and implementation.

## Rust Ownership Experiment

The Rust portion of the assignment uses a Jupyter notebook with the Rust kernel.

The notebook explores:

- basic Rust syntax
- mutable and immutable variables
- ownership and moving values
- cloning vectors
- borrowing values using references
- memory release when values go out of scope

The modified Rust notebook is included in this repository as `rust_vs_python_intro.ipynb`.

## Setup and Running Instructions

### 1. Clone the repository

```bash
git clone https://github.com/hanhan572/ids706-hw2.git
cd ids706-hw2
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

### 5. Run the Python data analysis

```bash
python data_analysis.py
```

Running the script will:

- inspect and clean the dataset
- perform filtering and grouping
- generate the wine quality visualization
- train and evaluate the Linear Regression model
- compare Pandas and Polars execution times

## Project Files

```text
ids706-hw2/
├── data/
│   └── wine_quality_merged.csv
├── data_analysis.py
├── README.md
├── requirements.txt
├── rust_vs_python_intro.ipynb
└── wine_quality_scatter.png
```

- `data_analysis.py` — Pandas and Polars analysis, visualization, and machine learning
- `data/wine_quality_merged.csv` — wine quality dataset
- `wine_quality_scatter.png` — visualization generated by the analysis
- `rust_vs_python_intro.ipynb` — modified Rust Jupyter notebook
- `requirements.txt` — Python package dependencies
- `README.md` — project documentation

## Tools Used

- Python
- Pandas
- Polars
- Matplotlib
- scikit-learn
- Rust
- Jupyter Notebook
- Git
- GitHub