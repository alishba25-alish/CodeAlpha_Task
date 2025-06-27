# 📊 Sales Prediction Using Advertising Data

This project predicts future product sales based on advertising budgets across different media platforms — **TV, Radio, and Newspaper** — using **Linear Regression**.

## 🎯 Objective

- Predict future sales using advertising spend as input features.
- Analyze how each medium (TV, Radio, Newspaper) impacts sales.
- Deliver actionable insights for marketing strategies.

## 🧹 Data Preparation

- Loaded dataset from `Advertising.csv`
- Dropped unnecessary column: `Unnamed: 0`
- Features used: `TV`, `Radio`, `Newspaper`
- Target variable: `Sales`

## 🤖 Model Used

- **Linear Regression** (from scikit-learn)
- Trained on 80% of the dataset
- Tested on the remaining 20%

## 📈 Model Evaluation

| Metric               | Value     |
|----------------------|-----------|
| Mean Squared Error   | ~3.17     |
| Mean Absolute Error  | ~1.46     |
| R² Score             | ~0.90     |

## 🔍 Insights from Model Coefficients

| Feature     | Coefficient |
|-------------|-------------|
| TV          | 0.0447      |
| Radio       | 0.1892      |
| Newspaper   | 0.0028      |

- **Radio** advertising has the strongest positive impact on sales.
- **TV** also contributes positively to sales.
- **Newspaper** has minimal impact on sales.

> 💡 *Businesses should focus more on **Radio** and **TV** advertising to increase sales, and may consider reducing **Newspaper** ad spending.*

## 📊 Actual vs Predicted Sales (Graph)

A scatter plot is used to visualize how closely the model's predictions match the actual sales values.

## 📉 Impact of Advertising Channels (Bar Chart)

A bar chart visualizes how strongly each channel (Radio, TV, Newspaper) influences sales, based on the regression model coefficients.
