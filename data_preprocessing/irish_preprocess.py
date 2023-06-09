from pathlib import Path

import numpy as np
import pandas as pd


def preprocess_irish_data(input_path: Path, output_path: Path):
    data_df = preprocess_data_df(input_path)
    data_df.to_pickle(output_path / "raw_data_df.pkl")

    allocation_df = preprocess_allocation_df(input_path)
    allocation_df.to_pickle(output_path / "raw_allocation_df.pkl")

    yearly_info_df = preprocess_pre_survey_df(input_path)
    yearly_info_df.to_pickle(output_path / "raw_yearly_info_df.pkl")


def preprocess_pre_survey_df(input_path):
    residential_files = [
        file
        for file in (
            input_path / "CER_Electricity_Data/Survey data - Excel format"
        ).iterdir()
        if "Residential" in str(file) and "pre" in str(file)
    ]
    survey_df = pd.read_excel(residential_files[0], index_col=0).replace({" ": np.nan})

    # select questions to keep
    questions = (
        pd.Series(survey_df.columns)
        .drop([0, 2, 3, 4, 5, 6, 7, 10])
        .drop(range(12, 33))
        .drop([34, 47, 56, 57])
        .drop(range(59, 114))
        .drop(range(116, 143))
    )

    # give the questions shorter names
    new_names = [
        "age",
        "household_composition",
        "number_of_adults",
        "number_of_children",
        "home_type",
        "build_year",
        "home_age",
        "floor_area",
        "floor_area_unit",
        "number_of_bedrooms",
        "heating_elec_central",
        "heating_elec_plugin",
        "heating_gas",
        "heating_oil",
        "heating_solid_fuel",
        "heating_renewable",
        "heating_other",
        "waterheating_central",
        "waterheating_elec_immersion",
        "waterheating_elec_instant",
        "waterheating_gas",
        "waterheating_oil",
        "waterheating_solid",
        "waterheating_renewable",
        "waterheating_other",
        "cooking",
        "BER_available",
        "BER_rating",
    ]

    yearly_info_df = (
        # select subset of questions
        survey_df.iloc[:, questions.index]
        # give them their new names
        .set_axis(new_names, axis=1)
        ## Attribute: age
        # convert age integers to meaningfull values (value denotes the start of the interval)
        .replace(dict(age={1: 18, 2: 26, 3: 36, 4: 46, 5: 56, 6: 65, 7: None}))
        # allows for int NA values (pd.NA)
        .astype(dict(age="Int32"))
        ## Attribute: household_composition, number_of_adults and number_of_children
        # people who are alone (household composition = 1)
        .mask(
            lambda x: x.household_composition == 1,
            lambda x: x.fillna({"number_of_adults": 1, "number_of_children": 0}),
        )
        # people with no children (age > 15)
        .mask(
            lambda x: x.household_composition == 2,
            lambda x: x.fillna({"number_of_children": 0}),
        )
        # people with children (no action needed)
        # household_composition no longer needed
        .drop(columns=["household_composition"])
        # cast to correct type
        .astype({"number_of_adults": "int8", "number_of_children": "int8"})
        ## Attribute: home_type
        .replace(
            dict(
                home_type={
                    1: "Apartment",
                    2: "Semi-detached",
                    3: "Detached",
                    4: "Terraced",
                    5: "Bungalow",
                    6: None,
                }
            )
        )
        ## Attribute: build_year
        .astype({"build_year": "Int16"})
        .mask(lambda x: x.build_year == 9999, lambda x: x.assign(build_year=None))
        .mask(lambda x: x.build_year < 1400, lambda x: x.assign(build_year=None))
        ## Attribute: home_age
        # only filled when previous question is 9999
        # calculated for others when build_year is known
        # + give some more interpreteable values
        .astype({"home_age": "Int8"})
        .mask(
            cond=lambda x: x.home_age.isna() & ~x.build_year.isna(),
            other=lambda x: x.assign(
                home_age=lambda df: age_in_years_to_category(2019 - df.build_year)
            ),
        )
        .replace(dict(home_age={1: 5, 2: 10, 3: 30, 4: 75, 5: 100}))
        ## Attribute: floor area, floor_area_unit
        .astype({"floor_area": "Int32"})
        .replace(dict(floor_area={999999999: pd.NA}))
        # if floor area unit == 2, convert from square feet to square meters
        .mask(
            lambda x: x.floor_area_unit == 2,
            lambda x: x.assign(floor_area=(x.floor_area / 10.764).astype("Int32")),
        )
        .drop(columns=["floor_area_unit"])
        ## Attribute number_of_bedrooms
        .astype({"number_of_bedrooms": "Int8"})
        .replace(dict(number_of_bedrooms={6: pd.NA}))
        ## Attribute all heating and water_heating attributes
        .pipe(
            lambda x: x.astype({col: "int8" for col in x.columns if "heating" in col})
        )
        ## Attribute: cooking
        .replace(dict(cooking={1: "Electric", 2: "Gas", 3: "Oil", 4: "Solid"}))
        ## Attribute: BER_available, BER_rating
        # only available for very few households
        .drop(columns=["BER_available", "BER_rating"])
    )

    return yearly_info_df


def preprocess_allocation_df(input_path):
    return (
        pd.read_excel(
            input_path
            / "CER_Electricity_Documentation/SME and Residential allocations.xlsx",
            dtype={"Id": "uint16", "Code": "uint8"},
            usecols=range(5),
        )
        .set_index("ID")
        .sort_index()
        .pipe(lambda x: x.set_axis(x.columns.str.lower(), axis=1))
        .assign(type=lambda x: x.code.replace({1: "Residential", 2: "SME", 3: "other"}))
        .drop(columns=["code"])
    )


def preprocess_data_df(input_path):
    data_files = list(input_path.glob("File*"))

    # read all data files
    dfs = [
        pd.read_csv(
            file_path,
            header=None,
            names=["id", "date_time", "load"],
            dtype={"id": "uint16", "load": "float32"},
            sep="\s+",
        )
        for file_path in data_files
    ]

    # add all individual dataframes
    df = pd.concat(dfs)

    # digit 1-3 (day 1 = 2019/1/1)
    day_code = df.date_time // 100 - 1

    # digit 4-5 (1 = 00:00:00 - 00:29:59)
    time_code = df.date_time % 100 - 1
    date = pd.to_datetime(
        day_code,
        unit="D",
        origin=pd.Timestamp("2009-01-01"),
        infer_datetime_format=True,
    )
    time_delta = pd.to_timedelta(time_code * 30, unit="m")

    # assign parsed date to date_time column
    df.date_time = date + time_delta

    # set index and sort
    df = df.set_index(["id", "date_time"]).sort_index()

    yearly_df = (
        df
        # keep last timestamps of duplicated hours
        .pipe(lambda x: x[~x.index.duplicated(keep="last")])
        # timestamps to columns
        .unstack()
        # drop first level of column multi-index
        .droplevel(0, axis=1)
        # make sure all timestamps are in the dataframe
        # need to transpose and detranspose because asfreq() cannot operate on columns
        .T.resample("30min", axis=0)
        .asfreq()
        .T
    )

    return yearly_df


def age_in_years_to_category(year_series):
    return (
        year_series.mask(lambda x: x <= 5, 1)
        .mask(lambda x: (x > 5) & (x <= 10), 2)
        .mask(lambda x: (x > 10) & (x <= 30), 3)
        .mask(lambda x: (x > 30) & (x <= 75), 4)
        .mask(lambda x: x > 75, 5)
    )
