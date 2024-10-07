import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ExploratoryDataAnalysis:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_features = ['Sqr Ft (sqft)', 'Beds', 'Bath', 'Lot Size (sqft)', 'Year Built']
        self.high_cardinality_cols = ['Price Sqr Ft ($/sqft)', 'Address', 'City', 'State', 'Address Full']

    def clean_data(self):
        self.clean_price_column()
        self.clean_lot_size_column()
        self.convert_numeric_features()
        if 'Price ($)' in self.df.columns:
            self.df['Log Price ($)'] = np.log1p(self.df['Price ($)'])

    def clean_price_column(self):
        if 'Price ($)' in self.df.columns:
            self.df['Price ($)'] = self.df['Price ($)'].replace('Contact For Estimate', np.nan)
            self.df['Price ($)'] = pd.to_numeric(self.df['Price ($)'], errors='coerce')

    def clean_lot_size_column(self):
        if 'Lot Size (sqft)' in self.df.columns:
            self.df['Lot Size (sqft)'] = self.df['Lot Size (sqft)'].apply(self.convert_lot_size)
            self.df['Lot Size (sqft)'] = pd.to_numeric(self.df['Lot Size (sqft)'], errors='coerce')

    @staticmethod
    def convert_lot_size(value):
        try:
            if pd.isna(value):
                return np.nan
            value_str = str(value).strip().lower()
            if 'acres' in value_str:
                number = float(value_str.replace('acres', '').strip())
                return number * 43560
            else:
                return float(value_str.replace(',', ''))
        except (ValueError, TypeError):
            return np.nan

    def convert_numeric_features(self):
        for feature in self.numeric_features:
            if feature in self.df.columns:
                self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')

    def plot_distribution(self, column_name, title, xlabel):
        if column_name in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[column_name].dropna(), bins=50, kde=True)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

    def plot_correlation_matrix(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Price ($)']
        if numeric_cols:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()

    def plot_scatter_plots(self):
        if 'Log Price ($)' in self.df.columns:
            valid_features = [feature for feature in self.numeric_features if feature in self.df.columns]
            num_plots = len(valid_features)
            rows = (num_plots + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
            axes = axes.flatten()
            for idx, feature in enumerate(valid_features):
                sns.scatterplot(data=self.df, x=feature, y='Log Price ($)', ax=axes[idx])
                axes[idx].set_title(f'{feature} vs. Log Price')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Log Price ($)')
            for idx in range(num_plots, len(axes)):
                fig.delaxes(axes[idx])
            plt.tight_layout()
            plt.show()

    def handle_high_cardinality_features(self):
        for col in self.high_cardinality_cols:
            if col in self.df.columns:
                if col in ['State', 'City']:
                    plt.figure(figsize=(12, 6))
                    avg_price = self.df.groupby(col)['Price ($)'].mean().nlargest(20)
                    avg_price.plot(kind='bar')
                    plt.title(f'Average House Price by {col}')
                    plt.xlabel(col)
                    plt.ylabel('Average Price ($)')
                    plt.tight_layout()
                    plt.show()

    def perform_eda(self):
        print("Starting EDA...")
        self.clean_data()
        print("Data cleaned successfully.")
        self.plot_distribution('Log Price ($)', 'Log-Transformed Distribution of House Prices', 'Log Price ($)')
        self.plot_correlation_matrix()
        self.plot_scatter_plots()
        self.handle_high_cardinality_features()
        print("EDA completed successfully.")

def model_price_prediction(train_data, test_data):
    X_train = train_data.drop("Price ($)", axis=1)
    y_train = train_data["Price ($)"]
    X_test = test_data.drop("Price ($)", axis=1)
    y_test = test_data["Price ($)"]

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train_scaled)
    y_pred_linear_scaled = linear_model.predict(X_test_scaled)
    y_pred_linear = scaler_y.inverse_transform(y_pred_linear_scaled.reshape(-1, 1)).flatten()
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    # Polynomial Regression
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train_scaled = poly_features.fit_transform(X_train_scaled)
    X_poly_test_scaled = poly_features.transform(X_test_scaled)
    poly_model = LinearRegression()
    poly_model.fit(X_poly_train_scaled, y_train_scaled)
    y_pred_poly_scaled = poly_model.predict(X_poly_test_scaled)
    y_pred_poly = scaler_y.inverse_transform(y_pred_poly_scaled.reshape(-1, 1)).flatten()
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)

    # SGD Regression
    sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    sgd_model.fit(X_train_scaled, y_train_scaled)
    y_pred_sgd_scaled = sgd_model.predict(X_test_scaled)
    y_pred_sgd = scaler_y.inverse_transform(y_pred_sgd_scaled.reshape(-1, 1)).flatten()
    mse_sgd = mean_squared_error(y_test, y_pred_sgd)
    r2_sgd = r2_score(y_test, y_pred_sgd)

    # Print results
    print(f"Linear Regression - MSE: {mse_linear:.4g}, R2: {r2_linear:.4g}")
    print(f"Polynomial Regression - MSE: {mse_poly:.4g}, R2: {r2_poly:.4g}")
    print(f"SGD Regression - MSE: {mse_sgd:.4g}, R2: {r2_sgd:.4g}")

    # Plot results
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.scatter(y_test, y_pred_linear)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Linear Regression')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

    plt.subplot(132)
    plt.scatter(y_test, y_pred_poly)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Polynomial Regression')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

    plt.subplot(133)
    plt.scatter(y_test, y_pred_sgd)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('SGD Regression')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

    plt.tight_layout()
    plt.show()

# Usage example
df = pd.read_csv(r'E:\Uni\2nd year\Intro to AI\Regression\assignment-1-ryantigi254\Regression\Weekly Tasks\cleaned_data.csv')
eda = ExploratoryDataAnalysis(df)
eda.perform_eda()

train_data = pd.read_csv(r'E:\Uni\2nd year\Intro to AI\Regression\assignment-1-ryantigi254\Regression\Weekly Tasks\train\train_data.csv')
test_data = pd.read_csv(r'E:\Uni\2nd year\Intro to AI\Regression\assignment-1-ryantigi254\Regression\Weekly Tasks\test\test_data.csv')
model_price_prediction(train_data, test_data)