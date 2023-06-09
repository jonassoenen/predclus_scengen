from pathlib import Path

import pandas as pd
import numpy as np

from day_info import get_daily_info_df_for_dates_and_city


def check_dataset(dataset):
    yearly_data_df, daily_data_df, yearly_info_df, daily_info_df = dataset

    # indices should be sorted
    assert all(
        df.index.is_monotonic_increasing
        for df in [yearly_data_df, daily_data_df, yearly_info_df, daily_info_df]
    ), "all indices should be sorted"

    # daily_data_df, yearly_info_df and daily_info_df cannot contain NaNs
    assert all(
        not df.isna().any().any()
        for df in [daily_data_df, yearly_info_df, daily_info_df]
    ), "dataframes cannot contain NaN"

    # index of daily and yearly dataframes should be identical
    assert all(
        yearly_data_df.index == yearly_info_df.index
    ), "yearly indices must be identical"

    if not all(daily_data_df.index == daily_info_df.index):
        uncommon_instances = daily_data_df.index.symmetric_difference(
            daily_info_df.index
        )
        if len(uncommon_instances) > 0:
            raise Exception(
                f"Some instances are in daily data df and not in daily info df or the other way around: \n {uncommon_instances}"
            )
        else:
            violating_instances = daily_data_df.index != daily_info_df.index
            viol_data = daily_data_df.index[violating_instances]
            viol_info = daily_info_df.index[violating_instances]
            raise Exception(
                f"Indices contain the same instances but their order is different \n{viol_data}\n{viol_info}"
            )

    # if meterID, date is in daily_data_df than meterID, year should be in yearly index
    alternative_index = daily_data_df.index.map(lambda x: (x[0], x[1].year))
    is_in_yearly_index = alternative_index.isin(yearly_data_df.index)
    if not all(is_in_yearly_index):
        violating_entries = alternative_index[~is_in_yearly_index].unique()
        raise Exception(
            f"Not all entries of the daily index are in the yearly index! \n These entries are not in the yearly index \n {violating_entries}"
        )
    assert all(
        daily_data_df.index.map(lambda x: (x[0], x[1].year)).isin(yearly_data_df.index)
    ), "if meterID, date is in daily_data_df meterID, year should be in yearly index"

    # if meterID, year is in yearly index than at least one meterID, date of that year should be in daily index
    assert all(
        yearly_data_df.index.isin(daily_data_df.index.map(lambda x: (x[0], x[1].year)))
    ), "if meterID, year is in yearly index than at least one meterID, date of that year should be in daily index"


def transform_raw_data_and_save(
    raw_data_df,
    yearly_info_df,
    result_path,
    weather_city,
    holiday_country,
    year_to_use_as_index=2012,
):
    if isinstance(raw_data_df, Path):
        raw_data_df = pd.read_pickle(raw_data_df)
    if isinstance(yearly_info_df, Path):
        yearly_info_df = pd.read_pickle(yearly_info_df)
    dfs = transform_raw_data_and_yearly_info(
        raw_data_df, yearly_info_df, weather_city, holiday_country, year_to_use_as_index
    )
    filenames = [
        f"{name}.pkl"
        for name in [
            "yearly_data_df",
            "daily_data_df",
            "yearly_info_df",
            "daily_info_df",
        ]
    ]
    result_path.mkdir(exist_ok=True)
    for filename, df in zip(filenames, dfs):
        df.to_pickle(result_path / filename)
    return dfs


def transform_raw_data_and_yearly_info(
    raw_data_df,
    yearly_info_df,
    weather_city,
    holiday_country,
    year_to_use_as_index=2012,
    missing_threshold=0.01,
):
    ## Construct the yearly data df
    if raw_data_df.index.nlevels == 1:
        # raw data contains single row for each profile, so split up in years
        yearly_data_df = raw_data_df_to_yearly_data_df(
            raw_data_df, year_to_use_as_index, missing_threshold
        )
    else:
        # raw data is already split per year
        yearly_data_df = raw_data_df.rename_axis(
            index=["meterID", "year"], columns="timestamp"
        )

    ## Construct yearly info df
    if yearly_info_df.index.nlevels == 1:
        # there is only a meterID index level! So reindex to match the yearly data

        # only retain profiles that are in both data and info
        desired_index = yearly_info_df.index.intersection(
            yearly_data_df.index.get_level_values(0)
        )
        yearly_data_df = yearly_data_df.loc[desired_index]
        # then reindex the yearly info df to add a second index level that matches the data df
        yearly_info_df = yearly_info_df.reindex(yearly_data_df.index, level=0)
    elif yearly_info_df.index.nlevels == 2:
        # only keep the entries that are in both yearly_info and yearly_data
        desired_index = yearly_info_df.index.intersection(yearly_data_df.index)
        yearly_info_df = yearly_info_df.loc[desired_index]
        yearly_data_df = yearly_data_df.loc[desired_index]
    elif yearly_info_df.index.nlevels > 2:
        raise Exception()
    # make sure axis is named correctly
    yearly_info_df = yearly_info_df.rename_axis(
        index=["meterID", "year"], columns="consumer_attributes"
    )

    # add yearly consumption as a feature
    yearly_info_df["yearly_consumption"] = yearly_data_df.sum(axis=1)

    ## Construct the daily data df
    if raw_data_df.index.nlevels == 1:
        # dataframe contains a single row for each profile
        daily_data_df = raw_data_df_to_daily_data_df(raw_data_df, year_to_use_as_index)
    else:
        # dataframe is already split per year
        daily_data_df = yearly_data_df_to_daily_data_df(raw_data_df)

    # daily data df now contains all the days, so drop the days that are not in the yearly df
    daily_data_df = daily_data_df.loc[
        # change daily_data index to (meterID, year)
        daily_data_df.index.map(lambda x: (x[0], x[1].year))
        # keep entries that are in the yearly index
        .isin(yearly_data_df.index)
    ]

    ## Construct daily info df
    dates = pd.to_datetime(daily_data_df.index.get_level_values(1).unique())
    daily_info_df = (
        # get the raw data
        get_daily_info_df_for_dates_and_city(dates, weather_city, holiday_country)
        # make sure it has the same index as daily_data_df (drop the correct days)
        .reindex(daily_data_df.index, level=1).rename_axis(
            index=["meterID", "date"], columns="daily_attributes"
        )
    )

    dataset = (
        yearly_data_df.astype("float64").sort_index(),
        daily_data_df.astype("float64").sort_index(),
        yearly_info_df.sort_index(),
        daily_info_df.sort_index(),
    )
    check_dataset(dataset)

    return dataset


def yearly_data_df_to_daily_data_df(data_df):
    year_to_use_as_index = data_df.columns[0].year
    all_dates = np.unique(data_df.columns.date)
    freq = data_df.columns[1] - data_df.columns[0]
    periods = int(pd.Timedelta(days=1) / freq)
    new_columns = pd.date_range(
        f"{year_to_use_as_index}-01-01 0:00", periods=periods, freq=freq
    )
    data = data_df.to_numpy().reshape((-1, periods))
    new_index = [
        (meterID, year, day) for (meterID, year) in data_df.index for day in all_dates
    ]
    daily_data_df = (
        pd.DataFrame(
            data, index=pd.MultiIndex.from_tuples(new_index), columns=new_columns
        )
        .dropna(how="any", axis=0)
        .pipe(
            lambda df: df.set_axis(
                df.index.map(lambda x: (x[0], x[1], x[2].replace(x[1]))), axis=0
            )
        )
        .droplevel(1)
        # .rename_axis(index = ['meterID', 'year', 'date'], columns = 'timestamp')
        .rename_axis(index=["meterID", "date"], columns="timestamp")
    )
    return daily_data_df


def raw_data_df_to_daily_data_df(data_df, year_to_use_as_index=2012):
    all_dates = np.unique(data_df.columns.date)
    all_profiles = data_df.index
    freq = data_df.columns[1] - data_df.columns[0]
    periods = int(pd.Timedelta(days=1) / freq)
    columns = pd.date_range(
        f"{year_to_use_as_index}-01-01 0:00", periods=periods, freq=freq
    )
    data = data_df.to_numpy().reshape((-1, periods))
    daily_data_df = pd.DataFrame(
        data,
        index=pd.MultiIndex.from_product(
            [all_profiles, all_dates], names=["meterID", "date"]
        ),
        columns=columns,
    )
    daily_data_df = daily_data_df.dropna(how="any", axis=0)
    daily_data_df = (
        daily_data_df
        # code to have meterID, year, date index
        # .assign(year = (daily_data_df.index.get_level_values(1).year))
        # .set_index('year', append = True)
        # .swaplevel(1,2)
        # .rename_axis(index = ['meterID', 'year', 'date'], columns = 'timestamp')
        .rename_axis(index=["meterID", "date"], columns="timestamp")
    )
    return daily_data_df


def raw_data_df_to_yearly_data_df(
    data_df, year_to_use_as_index=2012, missing_threshold=0.01
):
    freq = data_df.columns[1] - data_df.columns[0]
    new_index = pd.date_range(
        f"{year_to_use_as_index}-01-01 00:00",
        f"{year_to_use_as_index}-12-31 23:30",
        freq=freq,
    )
    return (
        # start from the data df
        data_df.T
        # process the years separately
        .groupby(lambda idx: idx.year)
        .apply(
            lambda df:
            # set the index such that dates are in 2009
            df.set_axis(df.index.map(lambda t: t.replace(year=year_to_use_as_index)))
            # reindex such that all dates in 2012 are in the index
            .reindex(new_index)
            # and transpose such that the dates are the columns and the profileIDs are the index
            .T
        )
        # only keep years which are mostly complete
        .loc[lambda df: df.isna().sum(axis=1) < df.shape[1] * missing_threshold]
        # correct indices and rename
        .swaplevel(0, 1)
        .rename_axis(index=["meterID", "year"], columns="timestamp")
        .sort_index()
    )
