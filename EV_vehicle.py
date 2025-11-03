
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "/electric_vehicles_spec_2025.csv.csv"

df = pd.read_csv(file_path)


print("✅ Dataset loaded successfully!\n")
print("Number of rows and columns:", df.shape)
print("\nColumn names:\n", df.columns.tolist())

# Show first 5 rows
print("\nSample data:")
display(df.head())

# ---------------------------------------------
# 📊 STEP 5: Basic information and summary
# ---------------------------------------------
print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
display(df.describe(include='all'))

# ---------------------------------------------
# ⚙️ STEP 1: Mount Drive and import libraries
# ---------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

# ---------------------------------------------
# 📂 STEP 2: Load the dataset
# ---------------------------------------------
file_path = "/electric_vehicles_spec_2025.csv.csv"
df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully!")
print("Shape before cleaning:", df.shape)


# 🧹 STEP 3: Clean column names

df.columns = (
    df.columns.str.strip()      # remove spaces
              .str.lower()      # lowercase for consistency
              .str.replace(' ', '_')
              .str.replace('[^a-zA-Z0-9_]', '', regex=True)
)
print("\nUpdated column names:")
print(df.columns.tolist())

# 🩺 STEP 4: Handle missing values

# View missing value counts
print("\nMissing values per column before cleaning:\n", df.isnull().sum())

# Example strategy:
# - For numeric columns: fill with median
# - For categorical/text columns: fill with 'Unknown'
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna("Unknown", inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum().sum())

# 🧾 STEP 5: Remove duplicates

initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"\nRemoved {initial_rows - df.shape[0]} duplicate rows")


# 🔢 STEP 6: Convert numeric columns
# Try converting key columns safely to numeric
numeric_cols = ['price_usd', 'battery_capacity_kwh', 'range_km', 'charging_time_hours', 'top_speed_kmh']

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('[^0-9.]', '', regex=True)   # remove text symbols
            .replace('', np.nan)
            .astype(float)
        )


# ⚖️ STEP 7: Standardize values (example)
# Remove unrealistic values (e.g., negative prices or ranges)
if 'price_usd' in df.columns:
    df = df[df['price_usd'] > 0]

if 'range_km' in df.columns:
    df = df[df['range_km'] > 0]


# 📊 STEP 8: Summary of cleaned data

print("\n✅ Dataset Cleaned Successfully!")
print("Shape after cleaning:", df.shape)
print("\nSample of cleaned data:")
display(df.head())

print("\nData types after cleaning:")
print(df.dtypes)


# 💾 STEP 9: Save cleaned dataset (updated path)
# ---------------------------------------------
clean_path = "/cleaned_electric_vehicles_spec_2025.csv"
df.to_csv(clean_path, index=False)

print(f"\n✅ Cleaned dataset successfully saved to: {clean_path}")


# ⚙️ STEP 1: Import libraries and load cleaned dataset
import pandas as pd

clean_path = "/cleaned_electric_vehicles_spec_2025.csv"
df = pd.read_csv(clean_path)

print("✅ Cleaned dataset loaded successfully!")
print("Shape:", df.shape)

# ------------------------------------------------------

print("\n📊 Dataset Overview:")
print(df.info())

print("\n📈 Statistical Summary (numeric columns):")
print(df.describe().T)


# 🏷️ STEP 3: Count unique EV brands and models

if 'brand' in df.columns and 'model' in df.columns:
    print("\nNumber of unique brands:", df['brand'].nunique())
    print("Number of unique models:", df['model'].nunique())

    top_brands = df['brand'].value_counts().head(10)
    print("\nTop 10 EV Brands:\n", top_brands)

# ------------------------------------------------------
# ⚡ STEP 4: Average Range and Price by Brand
# ------------------------------------------------------
if {'brand', 'range_km', 'price_usd'}.issubset(df.columns):
    avg_stats = df.groupby('brand')[['range_km', 'price_usd']].mean().sort_values('range_km', ascending=False)
    print("\n🔋 Average Range and Price by Brand:\n")
    print(avg_stats.head(10))

# ------------------------------------------------------
# 🚗 STEP 5: Find EVs with Maximum and Minimum Specs
# ------------------------------------------------------
if 'range_km' in df.columns:
    max_range_ev = df.loc[df['range_km'].idxmax()]
    min_range_ev = df.loc[df['range_km'].idxmin()]
    print("\n⚡ EV with Maximum Range:\n", max_range_ev)
    print("\n🐢 EV with Minimum Range:\n", min_range_ev)

if 'price_usd' in df.columns:
    most_expensive = df.loc[df['price_usd'].idxmax()]
    cheapest = df.loc[df['price_usd'].idxmin()]
    print("\n💰 Most Expensive EV:\n", most_expensive)
    print("\n💸 Cheapest EV:\n", cheapest)

# ------------------------------------------------------
# ⛽ STEP 6: Correlation Analysis (Range, Battery, Price)
# ------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

numeric_features = ['battery_capacity_kwh', 'range_km', 'price_usd']
existing_features = [col for col in numeric_features if col in df.columns]

if len(existing_features) >= 2:
    print("\n📈 Correlation Matrix:")
    corr = df[existing_features].corr()
    print(corr)

    plt.figure(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("🔗 Correlation Between EV Features")
    plt.show()

# ------------------------------------------------------
# 🧩 STEP 7: Price-to-Range Ratio (Value Efficiency)
# ------------------------------------------------------
if {'price_usd', 'range_km'}.issubset(df.columns):
    df['price_per_km'] = (df['price_usd'] / df['range_km']).round(2)
    print("\n💹 Price per km Efficiency (lower = better value):")
    print(df[['brand', 'model', 'price_usd', 'range_km', 'price_per_km']].sort_values('price_per_km').head(10))

# ------------------------------------------------------
# 🌍 STEP 8: Save enhanced dataset
# ------------------------------------------------------
enhanced_path = "/enhanced_electric_vehicles_spec_2025.csv"
df.to_csv(enhanced_path, index=False)
print(f"\n✅ Enhanced dataset saved to: {enhanced_path}")



