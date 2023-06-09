import pandas as pd
import holidays
from sklearn.preprocessing import OrdinalEncoder

from world_weather_online import (
    get_weather_data_for_city,
    preprocess_weather,
)


def get_daily_info_df_for_dates_and_city(dates, weather_city, holiday_country):
    calendar_df = get_calendar_info_for_dates(dates, holiday_country)

    years = dates.year.unique()
    start_year, end_year = min(years), max(years)
    weather_df = get_weather_data_for_city(
        weather_city, first_year=start_year, last_year=end_year
    ).pipe(preprocess_weather)

    return (
        pd.concat([weather_df, calendar_df], axis=1)
        .pipe(preprocess_day_info)
        .loc[dates]
    )


def preprocess_day_info(day_info):
    ORDINALS = ["season", "isHoliday", "isWeekend"]
    day_info[ORDINALS] = OrdinalEncoder().fit_transform(
        day_info[ORDINALS].astype("str")
    )
    return day_info


def get_calendar_info_for_dates(dates, country="Belgium"):
    day_df = pd.DataFrame(index=dates)

    # add standard date information
    day_df = day_df.assign(
        dayOfMonth=lambda x: x.index.day,
        month=lambda x: x.index.month,
        year=lambda x: x.index.year,
        dayOfWeek=lambda x: x.index.weekday,
        isWeekend=lambda x: x.index.day_of_week >= 5,
        dayOfYear=lambda x: x.index.day_of_year,
    )

    # add season info
    day_df["season"] = day_df.index.map(season_from_date)

    # add holiday info
    if country.lower() == "belgium":
        holiday = holidays.BE()
    elif country.lower() == "ireland":
        holiday = holidays.IE()
    elif country.lower() == "uk":
        holiday = holidays.GB()
    elif country.lower() == "england":
        holiday = holidays.country_holidays("GB", subdiv="England")
    else:
        raise Exception(f"Unknown country {country}")
    day_df["isHoliday"] = day_df.index.map(lambda date: date in holiday)
    return day_df


def season_from_date(date):
    month = date.month
    if 3 <= month <= 5:
        return "spring"
    elif 6 <= month <= 8:
        return "summer"
    elif 9 <= month <= 11:
        return "autumn"
    return "winter"
