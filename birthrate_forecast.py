import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


COUNTRY_CODE = "NGA"  # Nigeria. Use GBR for United Kingdom, IND for India, GHA for Ghana
INDICATOR = "SP.DYN.CBRT.IN"


def fetch_birth_rate(country_code):
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
        f"{INDICATOR}?format=json&per_page=20000"
    )

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()[1]

    records = []
    for item in data:
        if item["value"] is not None:
            records.append({
                "year": int(item["date"]),
                "birth_rate": float(item["value"])
            })

    df = pd.DataFrame(records)
    df = df.sort_values("year")
    return df


def forecast_next_10_years(df):
    X = df[["year"]]
    y = df["birth_rate"]

    model = LinearRegression()
    model.fit(X, y)

    last_year = df["year"].max()
    future_years = np.arange(last_year + 1, last_year + 11)

    future_df = pd.DataFrame({"year": future_years})
    future_df["predicted_birth_rate"] = model.predict(future_df[["year"]])

    return future_df


def plot_result(df, forecast_df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["year"], df["birth_rate"], marker="o", label="Historical Birth Rate")
    plt.plot(
        forecast_df["year"],
        forecast_df["predicted_birth_rate"],
        marker="x",
        label="10-Year Forecast"
    )

    plt.title("Birth Rate Forecast")
    plt.xlabel("Year")
    plt.ylabel("Birth Rate per 1,000 People")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    df = fetch_birth_rate(COUNTRY_CODE)

    print("Historical data:")
    print(df.tail())

    forecast_df = forecast_next_10_years(df)

    print("\n10-year forecast:")
    print(forecast_df)

    plot_result(df, forecast_df)