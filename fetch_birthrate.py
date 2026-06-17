from io import BytesIO
from zipfile import ZipFile

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        f"Missing dependency: {missing_package}. "
        "Install dependencies with: python -m pip install -r requirements.txt"
    ) from None


BIRTH_RATE_INDICATOR = "SP.DYN.CBRT.IN"
MALE_POPULATION_INDICATOR = "SP.POP.TOTL.MA.IN"
FEMALE_POPULATION_INDICATOR = "SP.POP.TOTL.FE.IN"
REQUEST_TIMEOUT_SECONDS = 15


def _records_to_dataframe(records, value_column):
    df = pd.DataFrame(records)

    if df.empty:
        raise Exception(f"No valid records found for {value_column}.")

    return df.sort_values("year")


def _fetch_from_download_csv(country_code, indicator, value_column):
    url = f"https://api.worldbank.org/v2/en/indicator/{indicator}?downloadformat=csv"

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not fetch {value_column} from the World Bank CSV download endpoint."
        ) from exc

    records = []

    with ZipFile(BytesIO(response.content)) as archive:
        data_file = next(
            name for name in archive.namelist()
            if name.startswith("API_") and name.endswith(".csv")
        )

        with archive.open(data_file) as csv_file:
            df = pd.read_csv(csv_file, skiprows=4)

    country_rows = df[df["Country Code"] == country_code]

    if country_rows.empty:
        raise Exception("No data found for this country.")

    row = country_rows.iloc[0]
    country = row["Country Name"]

    for year in df.columns[4:]:
        value = row[year]

        if pd.notna(value):
            records.append({
                "country": country,
                "country_code": country_code,
                "year": int(year),
                value_column: float(value),
            })

    return _records_to_dataframe(records, value_column)


def fetch_indicator(country_code, indicator, value_column):
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
        f"{indicator}?format=json&per_page=20000"
    )

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        try:
            json_data = response.json()
        except ValueError as exc:
            raise RuntimeError("World Bank API returned invalid JSON.") from exc
    except requests.exceptions.RequestException:
        try:
            return _fetch_from_download_csv(country_code, indicator, value_column)
        except Exception as csv_exc:
            raise RuntimeError(
                "Could not fetch data from the World Bank JSON API or CSV download endpoint. "
                "Check your internet connection and try again."
            ) from csv_exc

    if len(json_data) < 2 or json_data[1] is None:
        raise Exception("No data found for this country.")

    records = []

    for item in json_data[1]:
        year = item.get("date")
        value = item.get("value")
        country = item.get("country", {}).get("value")

        if value is not None:
            records.append({
                "country": country,
                "country_code": country_code,
                "year": int(year),
                value_column: float(value),
            })

    return _records_to_dataframe(records, value_column)


def fetch_birth_rate(country_code):
    """
    Fetch crude birth rate data from the World Bank API.
    Birth rate indicator: SP.DYN.CBRT.IN
    """
    return fetch_indicator(country_code, BIRTH_RATE_INDICATOR, "birth_rate")


def fetch_analysis_dataset(country_code):
    birth_rate = fetch_birth_rate(country_code)
    male_population = fetch_indicator(
        country_code,
        MALE_POPULATION_INDICATOR,
        "male_population",
    )
    female_population = fetch_indicator(
        country_code,
        FEMALE_POPULATION_INDICATOR,
        "female_population",
    )

    dataset = birth_rate.merge(
        male_population[["year", "male_population"]],
        on="year",
        how="inner",
    ).merge(
        female_population[["year", "female_population"]],
        on="year",
        how="inner",
    )

    dataset["male_to_female_rate"] = (
        dataset["male_population"] / dataset["female_population"] * 100
    )
    dataset["birth_rate_yearly_change"] = dataset["birth_rate"].diff()

    return dataset.sort_values("year")


def detect_birth_rate_outliers(dataset):
    q1 = dataset["birth_rate"].quantile(0.25)
    q3 = dataset["birth_rate"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = dataset[
        (dataset["birth_rate"] < lower_bound)
        | (dataset["birth_rate"] > upper_bound)
    ]

    return outliers, lower_bound, upper_bound


def explain_dataset(dataset, outliers, lower_bound, upper_bound):
    country = dataset["country"].iloc[0]
    first_year = int(dataset["year"].min())
    latest_year = int(dataset["year"].max())
    first_birth_rate = dataset.iloc[0]["birth_rate"]
    latest_birth_rate = dataset.iloc[-1]["birth_rate"]
    absolute_change = latest_birth_rate - first_birth_rate
    percent_change = absolute_change / first_birth_rate * 100
    latest_male_female_rate = dataset.iloc[-1]["male_to_female_rate"]

    print("\nDataset analysis")
    print("----------------")
    print(
        f"Country: {country}. The dataset is an annual World Bank time series "
        f"covering {first_year}-{latest_year}."
    )
    print(
        "Nature of data: each row represents one year. Columns include crude "
        "birth rate per 1,000 people, male population, female population, "
        "male-to-female population rate, and year-on-year birth-rate change."
    )
    print(f"Number of yearly records analysed: {len(dataset)}")
    print(
        f"Birth rate changed from {first_birth_rate:.2f} births per 1,000 people "
        f"in {first_year} to {latest_birth_rate:.2f} in {latest_year} "
        f"({absolute_change:.2f}, {percent_change:.1f}%)."
    )
    print(
        f"Latest male-to-female rate: {latest_male_female_rate:.2f} males per "
        "100 females. This is a population ratio, because the crude birth-rate "
        "indicator is not split into male and female births."
    )
    print(
        f"Outlier check: using the IQR method, possible outliers are birth-rate "
        f"values below {lower_bound:.2f} or above {upper_bound:.2f}."
    )

    if outliers.empty:
        print("No possible birth-rate outliers were detected.")
    else:
        print("Possible birth-rate outliers detected:")
        for _, row in outliers.iterrows():
            print(f"- {int(row['year'])}: {row['birth_rate']:.2f}")


def create_visualisation(dataset, outliers, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)

    axes[0].plot(
        dataset["year"],
        dataset["birth_rate"],
        color="#2563eb",
        marker="o",
        linewidth=2,
        label="Birth rate",
    )
    if not outliers.empty:
        axes[0].scatter(
            outliers["year"],
            outliers["birth_rate"],
            color="#dc2626",
            s=70,
            label="Possible outlier",
            zorder=3,
        )
    axes[0].set_title("Crude Birth Rate")
    axes[0].set_ylabel("Births per 1,000 people")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        dataset["year"],
        dataset["male_population"] / 1_000_000,
        color="#0f766e",
        linewidth=2,
        label="Male population",
    )
    axes[1].plot(
        dataset["year"],
        dataset["female_population"] / 1_000_000,
        color="#be123c",
        linewidth=2,
        label="Female population",
    )
    axes[1].set_title("Male and Female Population")
    axes[1].set_ylabel("Population, millions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        dataset["year"],
        dataset["male_to_female_rate"],
        color="#7c2d12",
        marker="o",
        linewidth=2,
        label="Males per 100 females",
    )
    axes[2].axhline(100, color="#6b7280", linestyle="--", linewidth=1)
    axes[2].set_title("Male-to-Female Population Rate")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Males per 100 females")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"{dataset['country'].iloc[0]} Birth Rate and Population Analysis",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    country_code = "NGA"  # Nigeria. Use GBR for United Kingdom, IND for India, GHA for Ghana.

    try:
        analysis_data = fetch_analysis_dataset(country_code)
        outliers, lower_bound, upper_bound = detect_birth_rate_outliers(analysis_data)

        csv_path = f"{country_code}_birth_rate_analysis.csv"
        image_path = f"{country_code}_birth_rate_analysis.png"

        analysis_data.to_csv(csv_path, index=False)
        explain_dataset(analysis_data, outliers, lower_bound, upper_bound)
        create_visualisation(analysis_data, outliers, image_path)

        print("\nRecent records:")
        print(analysis_data.tail(10))
        print(f"\nAnalysed dataset saved as {csv_path}")
        print(f"Visualisation saved as {image_path}")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from None
