from pathlib import Path

import pandas as pd
from secret import WEATHER_API_KEY
from wwo_hist import retrieve_hist_data


def read_weather_data(city):
    path = Path(__file__).absolute().parent / "weather_data" / f"weather_data_{city}.pkl"
    return pd.read_pickle(path)


def preprocess_weather(weather_df):
    weather_attributes = dict(
        maxtempC="maxTempC",
        mintempC="minTempC",
        tempC="avgTempC",
        sunHour="sunHour",
        uvIndex="uvIndex",
        FeelsLikeC="feelsLikeC",
        windspeedKmph="windspeedKmph",
        humidity="humidity",
        date_time="date_time",
    )
    return (
        weather_df[list(weather_attributes.keys())]
        .rename(columns=weather_attributes)
        .set_index("date_time")
        .astype("float")
    )


def get_weather_data_for_city(city, *, first_year=2011, last_year=2017, save=True):
    filepath = Path(__file__).parent / "data" / f"weather_data_{city.lower()}.pkl"
    if filepath.exists():
        return pd.read_pickle(filepath)
    all_days = pd.date_range(f"1/1/{first_year}", f"31/12/{last_year}", freq="1D")
    start_day = all_days.min().replace(month=1, day=1)
    end_day = all_days.max()

    end_dates = pd.date_range(start_day, end_day, freq="1M")
    start_dates = end_dates.map(lambda x: x.replace(day=1))

    intervals_to_get = list(zip(start_dates, end_dates))
    all_dfs = []
    for start_date, end_date in intervals_to_get:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        hist_weather_data = retrieve_hist_data(
            api_key=WEATHER_API_KEY,
            location_list=[city],
            start_date=start_str,
            end_date=end_str,
            frequency=24,
            export_csv=False,
            store_df=True,
        )[0]
        all_dfs.append(hist_weather_data)
    total_weather_data = pd.concat(all_dfs)
    if save:
        total_weather_data.to_pickle(filepath)
    return total_weather_data
