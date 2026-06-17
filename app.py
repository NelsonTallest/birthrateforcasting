from io import BytesIO, StringIO
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression


COUNTRY_COLUMN_HINTS = ("country", "country_name", "nation", "location", "area", "region", "description")
YEAR_COLUMN_HINTS = ("year", "financial_year", "date", "time", "period")
WORLD_BANK_HEADER_MARKERS = ("Country Name", "Country Code", "Indicator Name", "Indicator Code")
MAX_HEADER_SCAN_ROWS = 30
FORECAST_PERIODS = 5


def clean_dataset(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    df.columns = make_unique_columns(df.columns)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.drop_duplicates()

    for column in df.columns:
        if is_numeric_dtype(df[column]):
            continue

        cleaned = df[column].astype(str).str.strip()
        cleaned = cleaned.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
        numeric = pd.to_numeric(cleaned.str.replace(",", "", regex=False), errors="coerce")
        non_missing = cleaned.notna().sum()
        numeric_share = numeric.notna().sum() / non_missing if non_missing else 0
        if numeric_share >= 0.65 and numeric.notna().sum() >= 2:
            df[column] = numeric
        else:
            df[column] = cleaned

    return df


def make_unique_columns(columns):
    seen = {}
    unique_columns = []

    for column in columns:
        if column not in seen:
            seen[column] = 0
            unique_columns.append(column)
            continue

        seen[column] += 1
        unique_columns.append(f"{column}_{seen[column]}")

    return unique_columns


def normalize_year_label(value):
    text = str(value).strip().lower()
    text = re.sub(r"\.0$", "", text)
    text = text.replace("financial_year", "").replace("financial year", "").strip("_ -")

    if re.fullmatch(r"(18|19|20|21|22)\d{2}", text):
        year = int(text)
        if 1800 <= year <= 2200:
            return year

    match = re.fullmatch(r"((?:18|19|20|21)\d{2})[_\-/ ](\d{2})", text)
    if match:
        start_year = int(match.group(1))
        end_suffix = int(match.group(2))
        century = start_year // 100 * 100
        end_year = century + end_suffix
        if end_year < start_year:
            end_year += 100
        if 1800 <= end_year <= 2200:
            return end_year

    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return None

    date_like = re.search(
        r"\d{1,4}[-/]\d{1,2}|"
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)",
        text,
    )
    if not date_like:
        return None

    date = pd.to_datetime(value, errors="coerce")
    if pd.notna(date):
        return int(date.year)

    return None


def looks_like_year_label(value):
    return normalize_year_label(value) is not None


def extract_reporting_year(df):
    metadata_columns = [
        column for column in ("source_file", "source_sheet", "table_title")
        if column in df.columns
    ]

    for column in metadata_columns:
        values = df[column].dropna().astype(str).unique()
        for value in values:
            matches = re.findall(r"((?:18|19|20|21)\d{2}[_\-/ ]\d{2})", value)
            for match in matches:
                year = normalize_year_label(match)
                if year:
                    return year

    return None


def read_csv_with_header_detection(uploaded_file):
    content = uploaded_file.getvalue()
    text = content.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_row = 0

    for index, line in enumerate(lines[:20]):
        if all(marker in line for marker in WORLD_BANK_HEADER_MARKERS):
            header_row = index
            break

    return pd.read_csv(StringIO(text), skiprows=header_row)


def score_header_row(raw, row_index):
    row = raw.iloc[row_index]
    non_empty = row.notna().sum()
    if non_empty < 2:
        return -1

    year_like = sum(looks_like_year_label(value) for value in row.dropna())
    row_numeric = 0
    for value in row.dropna():
        if looks_like_year_label(value):
            continue
        numeric_value = pd.to_numeric(str(value).replace(",", ""), errors="coerce")
        if pd.notna(numeric_value):
            row_numeric += 1

    next_rows = raw.iloc[row_index + 1: row_index + 6]
    numeric_cells = 0

    for value in next_rows.to_numpy().ravel():
        if pd.isna(value):
            continue
        numeric_value = pd.to_numeric(str(value).replace(",", ""), errors="coerce")
        if pd.notna(numeric_value):
            numeric_cells += 1

    text_lengths = [
        len(str(value))
        for value in row.dropna()
        if not looks_like_year_label(value)
    ]
    long_title_penalty = 4 if text_lengths and max(text_lengths) > 80 and non_empty <= 3 else 0

    numeric_support = min(numeric_cells, non_empty * 3)
    return non_empty * 2 + year_like * 6 + numeric_support - row_numeric * 8 - long_title_penalty


def detect_header_row(raw):
    scan_limit = min(MAX_HEADER_SCAN_ROWS, len(raw))
    scored_rows = [
        (score_header_row(raw, row_index), row_index)
        for row_index in range(scan_limit)
    ]
    if not scored_rows:
        return 0

    best_score = max(score for score, _ in scored_rows)
    plausible_rows = [
        row_index
        for score, row_index in scored_rows
        if score >= 8 and score >= best_score - 10
    ]
    return min(plausible_rows) if plausible_rows else max(scored_rows)[1]


def first_text_value(raw):
    for value in raw.iloc[:MAX_HEADER_SCAN_ROWS].to_numpy().ravel():
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return None


def make_columns_from_header(header_values):
    columns = []

    for index, value in enumerate(header_values):
        text = "" if pd.isna(value) else str(value).strip()
        columns.append(text if text and text.lower() != "nan" else f"column_{index + 1}")

    return make_unique_columns(columns)


def read_excel_sheet(raw, sheet_name, file_name):
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        return None

    title = first_text_value(raw)
    header_row = detect_header_row(raw)
    table = raw.iloc[header_row + 1:].copy()
    table.columns = make_columns_from_header(raw.iloc[header_row])
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if table.empty:
        return None

    table["source_file"] = file_name
    table["source_sheet"] = sheet_name
    table["table_title"] = title or sheet_name
    return table


def read_excel_workbook(uploaded_file):
    workbook_bytes = uploaded_file.getvalue()
    excel_file = pd.ExcelFile(BytesIO(workbook_bytes))
    tables = []

    for sheet_name in excel_file.sheet_names:
        raw = pd.read_excel(
            BytesIO(workbook_bytes),
            sheet_name=sheet_name,
            header=None,
            dtype=object,
        )
        table = read_excel_sheet(raw, sheet_name, uploaded_file.name)
        if table is not None:
            tables.append(table)

    return tables


def load_uploaded_tables(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return [read_csv_with_header_detection(uploaded_file)]

    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return read_excel_workbook(uploaded_file)

    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")


def year_columns(df):
    return [
        column for column in df.columns
        if looks_like_year_label(column)
    ]


def reshape_world_bank_wide_data(df):
    years = year_columns(df)

    if not years:
        return df

    id_columns = [column for column in df.columns if column not in years]
    long_df = df.melt(
        id_vars=id_columns,
        value_vars=years,
        var_name="year_label",
        value_name="value",
    )
    long_df["year"] = long_df["year_label"].map(normalize_year_label)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"])

    if "indicator_name" in long_df.columns:
        long_df["outcome_name"] = long_df["indicator_name"]
    elif "table_title" in long_df.columns:
        long_df["outcome_name"] = long_df["table_title"]

    return long_df


def prepare_uploaded_dataset(uploaded_file):
    prepared_tables = []

    for raw_data in load_uploaded_tables(uploaded_file):
        cleaned = clean_dataset(raw_data)
        reshaped = reshape_world_bank_wide_data(cleaned)
        prepared_tables.append(reshaped)

    return prepared_tables


def merge_metadata(data, metadata_frames):
    if "country_code" not in data.columns or not metadata_frames:
        return data

    for metadata in metadata_frames:
        if "country_code" not in metadata.columns:
            continue

        metadata_columns = [
            column for column in metadata.columns
            if column == "country_code" or column not in data.columns
        ]
        data = data.merge(
            metadata[metadata_columns].drop_duplicates("country_code"),
            on="country_code",
            how="left",
        )

    return data


def find_column(columns, hints):
    normalized = list(columns)
    for hint in hints:
        for column in normalized:
            tokens = set(str(column).split("_"))
            if column == hint or hint in tokens:
                return column
    return None


def likely_location_column(df):
    hinted = find_column(df.columns, COUNTRY_COLUMN_HINTS)
    if hinted:
        return hinted

    candidates = []
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            continue

        values = df[column].dropna().astype(str).str.strip()
        unique_count = values.nunique()
        if 2 <= unique_count <= max(200, len(df) * 0.9):
            name_score = 2 if column in {"description", "column_2"} else 0
            candidates.append((name_score, unique_count, column))

    return sorted(candidates, reverse=True)[0][2] if candidates else None


def likely_time_column(df):
    for column in df.columns:
        column_text = str(column)
        tokens = set(column_text.split("_"))
        if column_text in YEAR_COLUMN_HINTS or tokens.intersection(YEAR_COLUMN_HINTS):
            coerced = coerce_year_values(df[column], allow_sequence=False)
            valid_count = coerced.notna().sum()
            if valid_count >= 2 and coerced.nunique(dropna=True) >= 2:
                return column

    return None


def numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def coerce_year_values(series, allow_sequence=True):
    normalized_years = series.map(normalize_year_label)
    if normalized_years.notna().mean() >= 0.6:
        return normalized_years

    numeric = pd.to_numeric(series, errors="coerce")
    plausible_years = numeric.between(1800, 2200)
    if numeric.notna().mean() >= 0.8 and plausible_years[numeric.notna()].mean() >= 0.8:
        return numeric
    if allow_sequence and numeric.notna().mean() >= 0.8:
        return numeric

    text_values = series.dropna().astype(str).str.strip()
    date_like = text_values.str.contains(
        r"\d{1,4}[-/]\d{1,2}|"
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)",
        case=False,
        regex=True,
    )
    if text_values.empty or date_like.mean() < 0.6:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")

    parsed_dates = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed_dates.dt.year


def likely_target_column(df, year_column):
    candidates = [column for column in numeric_columns(df) if column != year_column]
    if not candidates:
        return None

    scored = []
    for column in candidates:
        completeness = df[column].notna().mean()
        variance = df[column].nunique(dropna=True)
        scored.append((completeness, variance, column))

    return sorted(scored, reverse=True)[0][2]


def categorical_filter_columns(df):
    columns = []

    for column in df.columns:
        if is_numeric_dtype(df[column]):
            continue
        unique_count = df[column].dropna().astype(str).nunique()
        if 1 < unique_count <= 200:
            columns.append(column)

    preferred = [
        "outcome_name",
        "table_title",
        "source_sheet",
        "country",
        "country_name",
        "area",
        "column_1",
    ]
    return sorted(
        columns,
        key=lambda column: (
            preferred.index(column) if column in preferred else len(preferred),
            column,
        ),
    )


def detect_outliers(df, target_column):
    series = df[target_column].dropna()
    if series.empty:
        return df.iloc[0:0], None, None

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]

    return outliers, lower_bound, upper_bound


def fit_polynomial_model(x_values, y_values, degree):
    coefficients = np.polyfit(x_values, y_values, degree)
    predictions = np.polyval(coefficients, x_values)
    residual = np.sum((y_values - predictions) ** 2)
    total = np.sum((y_values - np.mean(y_values)) ** 2)
    score = 1 - residual / total if total else 0
    return coefficients, score


def assess_time_series_analysis(df, year_column, target_column):
    series_data = df[[year_column, target_column]].dropna().sort_values(year_column)
    series_data[year_column] = pd.to_numeric(series_data[year_column], errors="coerce")
    series_data[target_column] = pd.to_numeric(series_data[target_column], errors="coerce")
    series_data = series_data.dropna().groupby(year_column, as_index=False)[target_column].mean()

    if len(series_data) < 2:
        return {
            "recommended": False,
            "status": "Not enough ordered observations",
            "message": "Time series analysis is not suitable yet because at least two ordered records are required.",
        }

    unique_periods = series_data[year_column].nunique()
    target_variation = series_data[target_column].nunique()
    period_diffs = np.diff(np.sort(series_data[year_column].to_numpy(dtype=float)))
    regular_spacing = bool(
        len(period_diffs) == 0
        or np.allclose(period_diffs, np.median(period_diffs), rtol=0.05, atol=0.05)
    )

    if unique_periods >= 6 and target_variation >= 3 and regular_spacing:
        return {
            "recommended": True,
            "status": "Use time series analysis",
            "message": (
                "The selected data has enough regularly ordered periods and changing values, "
                "so trend-based time series analysis is appropriate for the five-year forecast."
            ),
        }

    if unique_periods >= 4 and target_variation >= 3:
        return {
            "recommended": True,
            "status": "Use with caution",
            "message": (
                "Time series analysis can be used, but the result should be treated carefully because "
                "the timeline is short or unevenly spaced."
            ),
        }

    return {
        "recommended": False,
        "status": "Prefer descriptive analysis",
        "message": (
            "The selected data has limited time depth or little outcome variation. "
            "A descriptive chart may be more reliable than a trend forecast."
        ),
    }


def forecast_next_five_years(df, year_column, target_column):
    model_data = df[[year_column, target_column]].dropna().sort_values(year_column)
    model_data[year_column] = model_data[year_column].astype(float)
    model_data = model_data.groupby(year_column, as_index=False)[target_column].mean()

    if model_data.empty:
        raise ValueError("No usable numeric rows are available for prediction.")

    if len(model_data) < 2:
        last_year = float(model_data[year_column].iloc[-1])
        last_value = float(model_data[target_column].iloc[-1])
        future_years = [last_year + offset for offset in range(1, FORECAST_PERIODS + 1)]
        forecast = pd.DataFrame({year_column: future_years})
        forecast[f"predicted_{target_column}"] = last_value
        return forecast, 0, "baseline projection"

    x = model_data[year_column].to_numpy(dtype=float)
    y = model_data[target_column].to_numpy(dtype=float)

    candidate_degrees = [1]
    if len(model_data) >= 4:
        candidate_degrees.append(2)
    if len(model_data) >= 7:
        candidate_degrees.append(3)

    fitted_models = []
    for degree in candidate_degrees:
        coefficients, score = fit_polynomial_model(x, y, degree)
        penalty = 0.015 * (degree - 1)
        fitted_models.append((score - penalty, score, degree, coefficients))

    _, r_squared, degree, coefficients = max(fitted_models, key=lambda item: item[0])

    step = np.median(np.diff(np.sort(x))) if len(x) > 1 else 1
    if not np.isfinite(step) or step <= 0:
        step = 1
    last_year = float(model_data[year_column].max())
    future_years = [last_year + step * offset for offset in range(1, FORECAST_PERIODS + 1)]
    forecast = pd.DataFrame({year_column: future_years})
    forecast[f"predicted_{target_column}"] = np.polyval(coefficients, future_years)
    model_name = {
        1: "linear trend",
        2: "quadratic trend",
        3: "cubic trend",
    }[degree]

    return forecast, r_squared, model_name


def describe_data(df, country_column, year_column, target_column):
    lines = []
    rows, columns = df.shape
    lines.append(f"The cleaned dataset contains {rows:,} rows and {columns:,} columns.")

    if country_column:
        countries = df[country_column].dropna().nunique()
        lines.append(f"It includes {countries:,} distinct country or location values.")

    if year_column:
        years = df[year_column].dropna()
        if not years.empty:
            min_year = int(years.min())
            max_year = int(years.max())
            lines.append(f"The time coverage runs from {min_year} to {max_year}.")
        else:
            lines.append("No valid year values were found in the selected year column.")

    if target_column:
        target = df[target_column].dropna()
        lines.append(
            f"The selected outcome, `{target_column}`, ranges from "
            f"{target.min():,.2f} to {target.max():,.2f}, with an average of {target.mean():,.2f}."
        )

    missing_cells = int(df.isna().sum().sum())
    lines.append(f"After basic cleaning, there are {missing_cells:,} missing cells remaining.")

    return " ".join(lines)


def plot_analysis(history, forecast, outliers, year_column, target_column):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        history[year_column],
        history[target_column],
        marker="o",
        color="#2563eb",
        linewidth=2,
        label="Historical data",
    )
    ax.plot(
        forecast[year_column],
        forecast[f"predicted_{target_column}"],
        marker="o",
        linestyle="--",
        color="#dc2626",
        linewidth=2,
        label="5-year forecast",
    )

    if not outliers.empty:
        ax.scatter(
            outliers[year_column],
            outliers[target_column],
            s=80,
            color="#f59e0b",
            label="Possible outlier",
            zorder=3,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(target_column.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_five_year_prediction(history, forecast, year_column, target_column):
    fig, ax = plt.subplots(figsize=(10, 5))
    history_data = (
        history[[year_column, target_column]]
        .dropna()
        .sort_values(year_column)
        .groupby(year_column, as_index=False)[target_column]
        .mean()
    )
    predicted_column = f"predicted_{target_column}"

    ax.plot(
        history_data[year_column],
        history_data[target_column],
        color="#2563eb",
        marker="o",
        linewidth=2,
        label="Historical outcome",
    )
    ax.plot(
        forecast[year_column],
        forecast[predicted_column],
        color="#16a34a",
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Predicted next five years",
    )

    if not history_data.empty and not forecast.empty:
        ax.plot(
            [history_data[year_column].iloc[-1], forecast[year_column].iloc[0]],
            [history_data[target_column].iloc[-1], forecast[predicted_column].iloc[0]],
            color="#94a3b8",
            linestyle=":",
            linewidth=1.5,
        )

    ax.set_title("Predicted Outcome in the Next Five Years")
    ax.set_xlabel(year_column.replace("_", " ").title())
    ax.set_ylabel(target_column.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_overview(df, target_column, group_column=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    if group_column and group_column in df.columns:
        summary = (
            df.dropna(subset=[group_column, target_column])
            .groupby(group_column)[target_column]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        if summary.empty:
            ax.text(0.5, 0.5, "No values available for this overview chart.", ha="center", va="center")
            ax.axis("off")
            return fig

        summary.sort_values().plot(kind="barh", ax=ax, color="#0f766e")
        ax.set_xlabel(target_column.replace("_", " ").title())
        ax.set_ylabel(group_column.replace("_", " ").title())
    else:
        values = df[target_column].dropna()
        if values.empty:
            ax.text(0.5, 0.5, "No values available for this overview chart.", ha="center", va="center")
            ax.axis("off")
            return fig

        values.plot(kind="hist", bins=20, ax=ax, color="#0f766e", alpha=0.85)
        ax.set_xlabel(target_column.replace("_", " ").title())
        ax.set_ylabel("Rows")

    ax.grid(True, axis="x", alpha=0.3)
    return fig


def plot_outlier_analysis(df, outliers, lower_bound, upper_bound, year_column, target_column):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = df[[year_column, target_column]].dropna().sort_values(year_column).copy()

    if plot_data.empty:
        ax.text(0.5, 0.5, "No values available for outlier visualisation.", ha="center", va="center")
        ax.axis("off")
        return fig

    normal_points = plot_data.drop(index=outliers.index, errors="ignore")
    ax.scatter(
        normal_points[year_column],
        normal_points[target_column],
        color="#2563eb",
        alpha=0.75,
        s=55,
        label="Normal value",
    )

    if not outliers.empty:
        ax.scatter(
            outliers[year_column],
            outliers[target_column],
            color="#dc2626",
            edgecolor="#7f1d1d",
            linewidth=0.8,
            s=95,
            label="Possible outlier",
            zorder=3,
        )

    if lower_bound is not None and upper_bound is not None:
        ax.axhline(lower_bound, color="#f59e0b", linestyle="--", linewidth=1.5, label="Lower IQR limit")
        ax.axhline(upper_bound, color="#f97316", linestyle="--", linewidth=1.5, label="Upper IQR limit")

    ax.set_title("Possible Outliers")
    ax.set_xlabel(year_column.replace("_", " ").title())
    ax.set_ylabel(target_column.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def dataframe_to_csv(df):
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def main():
    st.set_page_config(
        page_title="Dataset Analysis and Forecast",
        layout="wide",
    )
    st.title("Dataset Analysis and 5-Year Forecast")

    uploaded_files = st.file_uploader(
        "Upload country dataset files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more CSV or Excel datasets to begin.")
        st.stop()

    data_frames = []
    metadata_frames = []
    read_errors = []

    for uploaded_file in uploaded_files:
        try:
            prepared_tables = prepare_uploaded_dataset(uploaded_file)
        except Exception as exc:
            read_errors.append(f"{uploaded_file.name}: {exc}")
            continue

        for prepared in prepared_tables:
            if {"year", "value"}.issubset(prepared.columns):
                data_frames.append(prepared)
            elif "year" in prepared.columns and numeric_columns(prepared):
                data_frames.append(prepared)
            elif numeric_columns(prepared):
                data_frames.append(prepared)
            else:
                metadata_frames.append(prepared)

    if read_errors:
        st.warning("Some files could not be read:\n\n" + "\n".join(f"- {error}" for error in read_errors))

    if not data_frames:
        st.error(
            "No forecastable time-series dataset was found. "
            "Upload a dataset with a year/date column and at least one numeric outcome column. "
            "World Bank indicator CSV files are supported automatically."
        )
        if metadata_frames:
            st.subheader("Uploaded Metadata Preview")
            st.dataframe(metadata_frames[0].head(100), use_container_width=True)
        st.stop()

    data = pd.concat(data_frames, ignore_index=True, sort=False)
    data = merge_metadata(data, metadata_frames)

    if data.empty:
        st.error("The uploaded file does not contain any usable rows after cleaning.")
        st.stop()

    country_guess = likely_location_column(data)
    year_guess = likely_time_column(data)

    if year_guess:
        data[year_guess] = coerce_year_values(data[year_guess], allow_sequence=False)
        if data[year_guess].notna().sum() < 2 or data[year_guess].nunique(dropna=True) < 2:
            year_guess = None

    if not year_guess:
        reporting_year = extract_reporting_year(data)
        if reporting_year:
            data["reporting_year"] = reporting_year
            year_guess = "reporting_year"
        else:
            data["record_index"] = np.arange(1, len(data) + 1)
            year_guess = "record_index"

    target_guess = likely_target_column(data, year_guess)
    filter_columns = categorical_filter_columns(data)

    with st.sidebar:
        st.header("Analysis Setup")
        country_column = st.selectbox(
            "Country column",
            ["None"] + data.columns.tolist(),
            index=(data.columns.tolist().index(country_guess) + 1) if country_guess else 0,
        )
        year_column = st.selectbox(
            "Year column",
            data.columns.tolist(),
            index=data.columns.tolist().index(year_guess) if year_guess else 0,
        )

        data[year_column] = coerce_year_values(
            data[year_column],
            allow_sequence=year_column == "record_index",
        )
        available_targets = [column for column in numeric_columns(data) if column != year_column]

        if not available_targets:
            st.error("No numeric outcome column was found.")
            st.stop()

        target_column = st.selectbox(
            "Outcome to predict",
            available_targets,
            index=available_targets.index(target_guess) if target_guess in available_targets else 0,
        )

        selected_country = None
        if country_column != "None":
            countries = sorted(data[country_column].dropna().astype(str).unique())
            if not countries:
                st.error("The selected country column has no usable country values.")
                st.stop()
            selected_country = st.selectbox("Country/location to analyse", ["All locations"] + countries)

        selected_indicator = None
        if "indicator_name" in data.columns:
            indicators = sorted(data["indicator_name"].dropna().astype(str).unique())
            if len(indicators) > 1:
                selected_indicator = st.selectbox("Indicator to forecast", indicators)

        filter_column = st.selectbox(
            "Series or sheet filter",
            ["None"] + filter_columns,
            index=(filter_columns.index("outcome_name") + 1) if "outcome_name" in filter_columns else 0,
        )
        selected_filter_value = None
        if filter_column != "None":
            filter_values = sorted(data[filter_column].dropna().astype(str).unique())
            selected_filter_value = st.selectbox("Filter value", filter_values)

    country_data = data.copy()
    if (
        country_column != "None"
        and selected_country is not None
        and selected_country != "All locations"
    ):
        country_data = country_data[country_data[country_column].astype(str) == selected_country]
    if selected_indicator is not None:
        country_data = country_data[country_data["indicator_name"].astype(str) == selected_indicator]
    if selected_filter_value is not None:
        country_data = country_data[country_data[filter_column].astype(str) == selected_filter_value]

    country_data = country_data.dropna(subset=[year_column, target_column])
    country_data = country_data.sort_values(year_column)

    st.subheader("Data Description")
    st.write(describe_data(data, None if country_column == "None" else country_column, year_column, target_column))

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(data):,}")
    metric_columns[1].metric("Columns", f"{len(data.columns):,}")
    metric_columns[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    metric_columns[3].metric("Forecast rows", f"{len(country_data):,}")

    outliers, lower_bound, upper_bound = detect_outliers(country_data, target_column)
    time_series_recommendation = assess_time_series_analysis(country_data, year_column, target_column)

    try:
        forecast, r_squared, model_name = forecast_next_five_years(country_data, year_column, target_column)
    except Exception as exc:
        st.error(f"Could not create forecast: {exc}")
        st.stop()

    st.subheader("Forecast")
    if country_column != "None" and selected_country is not None:
        st.write(f"Forecasting `{target_column}` for `{selected_country}` for the nearest five years.")
    else:
        st.write(f"Forecasting `{target_column}` for the uploaded dataset for the nearest five years.")
    if model_name == "baseline projection":
        st.write(
            "Model used: baseline projection. A trend-based time-series model could not be fitted "
            "because the selected data has only one reporting period."
        )
    else:
        st.write(f"Model used: best-fit {model_name}. Fit score on historical data: {r_squared:.3f}.")

    st.subheader("Time Series Analysis Recommendation")
    if time_series_recommendation["recommended"]:
        st.success(f"{time_series_recommendation['status']}: {time_series_recommendation['message']}")
    else:
        st.warning(f"{time_series_recommendation['status']}: {time_series_recommendation['message']}")

    st.pyplot(plot_analysis(country_data, forecast, outliers, year_column, target_column))

    st.subheader("Next Five Years Prediction Visualisation")
    st.pyplot(plot_five_year_prediction(country_data, forecast, year_column, target_column))

    st.subheader("Best-Fit Data Visualisation")
    overview_group = None
    if selected_filter_value is None and filter_columns:
        overview_group = filter_columns[0]
    elif country_column != "None":
        overview_group = country_column
    st.pyplot(plot_overview(country_data, target_column, overview_group))

    left, right = st.columns(2)
    with left:
        st.subheader("Possible Outliers")
        st.pyplot(
            plot_outlier_analysis(
                country_data,
                outliers,
                lower_bound,
                upper_bound,
                year_column,
                target_column,
            )
        )
        if outliers.empty:
            st.write("No possible outliers were detected using the IQR method.")
        else:
            st.write(
                f"Values below {lower_bound:,.2f} or above {upper_bound:,.2f} "
                "are flagged as possible outliers."
            )
            st.dataframe(outliers, use_container_width=True)

    with right:
        st.subheader("5-Year Prediction")
        st.dataframe(forecast, use_container_width=True)

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(data.head(100), use_container_width=True)

    st.download_button(
        "Download cleaned dataset",
        data=dataframe_to_csv(data),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download forecast",
        data=dataframe_to_csv(forecast),
        file_name="forecast_next_5_years.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
