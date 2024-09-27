# Video Game Sales Data Analysis

## Project Overview

This project was completed as part of the Exploratory Data Analysis course during the year 2020. The primary objective is to analyze a dataset containing sales information for over 100,000 video games and perform both supervised and unsupervised machine learning tasks. The project is divided into four main sections: data processing, supervised learning (classification and regression), dimensionality reduction, and clustering. Finally, a report is produced answering key questions related to the analysis.

The project is implemented using Python, leveraging libraries such as Pandas, Scikit-learn, and Altair for data processing, machine learning, and visualization.

## Objectives

- Understand, analyze, and process a given dataset.
- Select the most relevant features for solving the classification and regression problems.
- Implement supervised learning models (classification and regression) and evaluate their performance.
- Gain familiarity with unsupervised learning techniques (clustering) and analyze their limitations.
- Visualize the results using Altair.
Dataset

The dataset used in this project (games.csv) contains information about various video games, with columns such as:

- ID: Video game identifier.
- Rank: Video game ranking.
- Name: Video game name.
- Platform: Platform used to play the game.
- Year: Year of release.
- Genre: Genre of the video game.
- Difficulty: Difficulty level of the game (values from 1 to 10).
- Publisher: Game publisher.
- IGN Score: Video game rating (values from 1 to 10).
- NA_Sales: Sales in North America (in millions).
- EU_Sales: Sales in Europe (in millions).
- JP_Sales: Sales in Japan (in millions).
- Other_Sales: Sales in other parts of the world (in millions).

## Project Structure

### Part 1: Data Processing

In this part, the dataset is loaded and cleaned using Pandas. The steps involved are:

1. Feature Selection: Irrelevant columns that do not contribute to the task are removed.
2. Handling Missing Values: Missing values are identified and handled either by imputing or removing them.
3. Data Normalization: The dataset is normalized to ensure uniformity across features. Two class labels are used: Publisher and Other_Sales (sales in the rest of the world).

### Part 2: Supervised Learning (Classification and Regression)

This section focuses on building and evaluating machine learning models for classification and regression:

1. Classification using KNN: The K-Nearest Neighbors (KNN) algorithm is applied to predict the publisher of the video games. The dataset is split into training and test sets to evaluate performance.
2. Performance metrics include accuracy, precision, recall, and F1-score.
3. Regression using Linear Regression: A Linear Regression model is applied to predict sales in other parts of the world. The performance is measured using Mean Squared Error (MSE).

Both models are tuned by adjusting hyperparameters, and the best configuration is selected based on performance.

### Part 3: Dimensionality Reduction and Clustering

In this part, the goal is to reduce the dataset's dimensionality and cluster the data:

1. Dimensionality Reduction: The T-distributed Stochastic Neighbor Embedding (t-SNE) technique is used to reduce the feature space to two dimensions, allowing for easier visualization.

2. Clustering: After reducing dimensionality, two clustering algorithms are applied:

A centroid-based algorithm (e.g., K-Means).
A density-based algorithm (e.g., DBSCAN).
3. Visualization: The clustered data is visualized in two dimensions using Altair.

### Part 4: Report and Analysis

The final section consists of answering a series of questions based on the analysis and results from the previous sections. The questions cover topics such as:

- Feature selection and handling of missing data.
- Rationale behind data normalization.
- Justification for splitting data into training and test sets.
- Hyperparameter tuning and its effects on model performance.
- The use of dimensionality reduction and clustering techniques, and their visualizations.

### Bonus: Alternative Models

As a bonus task, an additional classification and regression model is implemented (different from KNN and Linear Regression) to compare their performance with the original models.

## How to Run the Project

1. Clone the Repository:

```
git clone https://github.com/your-repository-url.git
cd your-repository-url
```

2. Install the Required Libraries (Python 3.x)

3. Run the Jupyter Notebook: You can either run the Jupyter notebook locally or use Google Colab:

Locally: Run the following command to start Jupyter Notebook:

```
jupyter notebook
```

Google Colab: Upload the task.ipynb notebook to Colab, and ensure the dataset games.csv is uploaded to the same environment.

### Example of Expected Output

When running the notebook, you can expect the following outputs:

- Cleaned dataset after preprocessing.
- Accuracy, precision, recall, and F1-score for the KNN classifier.
- Mean Squared Error (MSE) for the Linear Regression model.
- Two-dimensional scatter plots visualizing the t-SNE transformation and clustering results.
- A report answering key questions regarding data processing, model selection, and analysis.

## Results and Conclusion

The project demonstrates the use of supervised and unsupervised learning techniques to analyze a dataset of video game sales. By preprocessing the data, training machine learning models, and visualizing results, we gain insights into the performance of different algorithms and the relationships within the data.
