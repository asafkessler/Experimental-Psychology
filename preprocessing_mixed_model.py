import pandas as pd
import re
import os

RAW_DATA_PATH = "data/RawData_Only.xlsx"
GROUP_SCHEMA_PATH = "data/Group_Scenario_Numbers_and_Versions.csv"
OUTPUT_PATH = "data/long_format_mixed_model.csv"

os.makedirs("data", exist_ok=True)

# 60 data columns split into 6 groups of 10 (counterbalancing blocks)
GROUPS = ["A", "B", "C", "D", "E", "F"]


def assign_group(df: pd.DataFrame) -> pd.Series:
    """Return a Series mapping each row to its counterbalancing group (A-F)."""
    data_cols = list(df.columns[1:])
    group_series = pd.Series(index=df.index, dtype=str)
    for i, grp in enumerate(GROUPS):
        block = data_cols[i * 10 : (i + 1) * 10]
        mask = df[block].notna().any(axis=1)
        group_series[mask] = grp
    return group_series


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format raw data to long format for mixed models.

    Input columns: 'Response ID', '<scenario> positive/negative[.N]', ...
    Output columns: participant_id, group, scenario, framing, framing_dummy, rating
    """
    groups = assign_group(df)
    rows = []
    for idx, row in df.iterrows():
        pid = int(row["Response ID"])
        grp = groups[idx]
        for col in df.columns[1:]:
            val = row[col]
            if pd.isna(val):
                continue
            m = re.match(r"^(\d+)\s+(positive|negative)", col.strip())
            if m:
                scenario = int(m.group(1))
                framing = m.group(2)
                rows.append(
                    {
                        "participant_id": pid,
                        "group": grp,
                        "scenario": scenario,
                        "framing": framing,
                        "framing_dummy": 1 if framing == "positive" else 0,
                        "rating": float(val),
                    }
                )

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(["participant_id", "scenario"]).reset_index(drop=True)
    return long_df


def main():
    df = pd.read_excel(RAW_DATA_PATH)
    print(f"Loaded {len(df)} participants from {RAW_DATA_PATH}")

    long_df = wide_to_long(df)

    print(f"Long format shape: {long_df.shape}")
    print(f"Participants: {long_df['participant_id'].nunique()}")
    print(f"Scenarios: {sorted(long_df['scenario'].unique())}")
    print(f"Framing distribution:\n{long_df['framing'].value_counts()}")
    print(f"Group distribution:\n{long_df.groupby('group')['participant_id'].nunique()}")
    print(f"\nSample:\n{long_df.head(20).to_string()}")

    long_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
