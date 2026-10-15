
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/retail_sales_eda_dataset.csv")

# 1. Inspect the data
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# 2. Check data quality
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 3. Descriptive statistics
print("\nStatistical summary:")
print(df.describe())

# 4. Sales by product
sales_by_product = df.groupby("Product")["Sales_INR"].sum()
print("\nSales by product:")
print(sales_by_product.sort_values(ascending=False))

# 5. Create a chart
sales_by_product.sort_values().plot(
    kind="barh",
    figsize=(9, 5),
    title="Total Sales by Product"
)

plt.xlabel("Sales (INR)")
plt.ylabel("Product")
plt.tight_layout()
plt.show()
