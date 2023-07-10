"""
The main goal of this code is to process and transform data from multiple Excel files into
a comprehensive behavioral analysis matrix. The primary steps include reading the Excel files,
cleaning the data, calculating behavior statistics, pivoting data, and merging the resulting DataFrames.
"""

import functools as ft
import os

import numpy as np
import pandas as pd

# Requirements:
# out of sight (stop-start) aftrekken van observation duration
# end of observation (stop-start) aftrekken van observation duration

# v = ["First closest neighbour", "Second closest neighbour", "Group size", "Social calls", "End of Observation", "Out of Sight"]
# v mag weg uit de behavior times (hoeft niet in die matrix)
# elke behavior_run_time en behavior_count delen door observation duration

# uitzonderingen:
# u = ["First closest neighbour", "Second closest neighbour", "Group size", "Social calls"]
# u[0] en u[1] hebben een waarde van 1 tot >10 op Modifier #1 en een waarde van ["Certain", "Uncertain"] op Modifier #2
#   dit moet gewoon achter mekaar geplakt
# u[2] alleen de waarde van Modifier #1 pakken (als het goed is komt u[2] altijd 1x voor per observatie)
# u[3] opdelen in u[3] high en u[3] low. waarde van low staat aangegeven in Comment kolom. als Comment leeg: ga uit van LOW


# Specify the output file path for the intermediary full dataset and processed data
intermediate_output_file = os.getcwd() + "/data/processed/all_observations.csv"
processed_out_file = "data/processed/full_raven_behavior_matrix.xlsx"

# Specify the directory path containing the files
path = os.getcwd() + "/data/raw"

# Get the list of files in the directory with extension "xlsx"
files = os.listdir(path)
files_xls = [f"{path}/{f}" for f in files if f[-4:] == "xlsx"]

# Initialize an empty DataFrame
df = pd.DataFrame()

# Iterate through the files and concatenate data to the DataFrame
for f in files_xls:
    # Read data from each Excel file using the "openpyxl" engine
    data = pd.read_excel(f, 0, engine="openpyxl")
    # Concatenate the data to the existing DataFrame
    df = pd.concat([df, data], axis=0)

# Save the concatenated DataFrame to a CSV file
df.to_csv(intermediate_output_file, index=False, sep=";")


# append given/received to behavior, and do groupby aggregations again
condition = df["Modifier #1"].isin(["Given", "Received"])
df.loc[condition, "Behavior"] = (
    df.loc[condition, "Behavior"] + " - " + df.loc[condition, "Modifier #1"]
)


# Get state behavior start times
behavior_start_times = (
    df.loc[
        df["Behavior type"] == "START",
        ["Observation id", "Behavior", "Time"],
    ]
    .groupby(["Observation id", "Behavior"])
    .sum()
    .rename(columns={"Time": "start_time"})
)

# Get state behavior stop times
behavior_stop_times = (
    df.loc[
        df["Behavior type"] == "STOP",
        ["Observation id", "Behavior", "Time"],
    ]
    .groupby(["Observation id", "Behavior"])
    .sum()
    .rename(columns={"Time": "stop_time"})
)

# Combine start and stop times into behavior times DataFrame
behavior_times = behavior_start_times.join(behavior_stop_times)
behavior_times["run_time"] = behavior_times["stop_time"] - behavior_times["start_time"]
behavior_times = behavior_times.reset_index()

# Use info from ["Out of Sight", "End of Observation"] state behaviors to correct 'Observation duration'
observation_duration_correction_values = (
    behavior_times.loc[
        behavior_times["Behavior"].isin(["Out of Sight", "End of Observation"]),
        ["Observation id", "Behavior", "run_time"],
    ]
    .groupby(["Observation id", "Behavior"])
    .tail()
    .groupby("Observation id")
    .sum()[["run_time"]]
)
observation_duration_correction_values.rename(
    columns={"run_time": "observation_duration_correction_values"}, inplace=True
)

# Join correction values with the main DataFrame
df = df.join(observation_duration_correction_values, on="Observation id")

# Subtract correction values from 'Observation duration'
df["Observation duration"] = df["Observation duration"] - (
    df["observation_duration_correction_values"] + 0.001
)

# Drop the temporary column used for correction values
df = df.drop(columns=["observation_duration_correction_values"])


# exclude behaviors that are actually either observation statistics or markers for the data analyst
behavior_times = behavior_times.loc[
    ~behavior_times["Behavior"].isin(
        [
            "First closest neighbour",
            "Second closest neighbour",
            "Group size",
            "Social calls",
            "End of Observation",
            "Out of Sight",
        ]
    )
]

# add 'Observation duration' as a column
behavior_times = behavior_times.join(
    df[["Observation id", "Observation duration"]]
    .groupby("Observation id")
    .tail()
    .set_index("Observation id"),
    on="Observation id",
).drop_duplicates()
# calculate 'behavior_ratio' using 'run_time' and 'Observation duration'
behavior_times["behavior_ratio"] = behavior_times[
    "run_time"
]  # / behavior_times["Observation duration"]
# transform/pivot the dataframe into the required matrix for analysis in R
state_behavior_matrix = (
    behavior_times.pivot(
        index="Observation id", columns="Behavior", values="behavior_ratio"
    )
    .reset_index()
    .fillna(0)
)


# get event behavior stats
df["count"] = 1
behavior_counts = (
    df.loc[
        (df["Behavior type"] == "POINT")
        & (
            ~df["Behavior"].isin(
                [
                    "First closest neighbour",
                    "Second closest neighbour",
                    "Group size",
                    "Social calls",
                    "End of Observation",
                    "Out of Sight",
                ]
            )
        ),
        ["Observation id", "Observation duration", "Behavior", "count"],
    ]
    .groupby(["Observation id", "Behavior"])
    .sum()
)
# calculate 'behavior_frequency_rate' using 'count' and 'Observation duration'
behavior_counts["behavior_frequency_rate"] = behavior_counts[
    "count"
]  # / behavior_counts["Observation duration"]
# Create event behavior matrix
event_behavior_matrix = (
    behavior_counts.reset_index()
    .pivot(index="Observation id", columns="Behavior", values="behavior_frequency_rate")
    .reset_index()
    .fillna(0)
)


# Select rows with specific behaviors and columns of interest
observation_neighbor_stats = df.loc[
    df["Behavior"].isin(["First closest neighbour", "Second closest neighbour"]),
    ["Observation id", "Behavior", "Modifier #1"],
]

# Replace values in "Modifier #1" column and convert to integer
observation_neighbor_stats["Modifier #1"] = (
    observation_neighbor_stats["Modifier #1"]
    .replace({">10": 10})
    .astype(float)
    .astype(int)
)

# Create "neighbor_matrix" using pivot_table function
neighbor_matrix = pd.pivot_table(
    observation_neighbor_stats,
    index="Observation id",
    columns="Behavior",
    values="Modifier #1",
    aggfunc=np.min,
).fillna(0)


# Select rows with "Behavior" as "Group size" and specific columns
group_size_per_observation = df.loc[
    (df["Behavior"].isin(["Group size"])),
    [
        "Observation id",
        "Behavior",
        "Modifier #1",
    ],
]
# Pivot the DataFrame to create "group_size_per_observation" DataFrame
group_size_per_observation = group_size_per_observation.pivot(
    index="Observation id", columns="Behavior", values="Modifier #1"
)


# Extract "Social calls - High" and "Social calls - Low" values from "Comment" column
extracted_values = df.loc[
    df["Behavior"] == "Social calls",
    "Comment",
].str.extract(r"High: (\d+)\nLow: (\d+)", expand=True)

# Assign the extracted values to the columns
df.loc[df["Behavior"] == "Social calls", "Social calls - High"] = extracted_values[0]
df.loc[df["Behavior"] == "Social calls", "Social calls - Low"] = extracted_values[1]

# Mask to identify rows where "Behavior" is "Social calls" and "Comment" is NaN
social_calls_mask = (df["Behavior"] == "Social calls") & df["Comment"].isna()

# Fill "Social calls - Low" column with "Modifier #1" values for rows matching the mask
df.loc[social_calls_mask, "Social calls - Low"] = df.loc[
    social_calls_mask, "Modifier #1"
]

# Fill missing values in "Social calls - High" column with 0
df["Social calls - High"] = df["Social calls - High"].fillna(0)

# Create "social_calls_per_observation" with stats about social calls for each observation
social_calls_per_observation = df.loc[
    df["Behavior"] == "Social calls",
    [
        "Observation id",
        # Other columns that can be included as needed
        "Social calls - High",
        "Social calls - Low",
    ],
]


# Get `Observation duration` for each observation
duration_per_observation = df[
    ["Observation id", "Observation duration"]
].drop_duplicates()


# List of DataFrames to be joined
dfs = [
    state_behavior_matrix,
    neighbor_matrix,
    event_behavior_matrix,
    social_calls_per_observation,
    group_size_per_observation,
    duration_per_observation,
]
# Left join DataFrames recursively on `Observation id` using reduce and merge operations
full_raven_behavior_matrix = ft.reduce(
    lambda left, right: pd.merge(left, right, on="Observation id", how="left"), dfs
)
full_raven_behavior_matrix = full_raven_behavior_matrix.fillna(0)
full_raven_behavior_matrix = (
    full_raven_behavior_matrix.set_index("Observation id").astype(float).reset_index()
)


full_raven_behavior_matrix.to_excel(
    processed_out_file,
    index=False,
)

# Final prints to preview the results (of this script)
print(f"Here's a preview of the full raven behavior matrix:")
print(full_raven_behavior_matrix.head())
print()
print(f"Find the processed output file in: `{processed_out_file}`")
