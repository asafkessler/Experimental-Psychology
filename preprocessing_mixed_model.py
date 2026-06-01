import pandas as pd
import re
import os

RAW_DATA_PATH = "data/RawData_Only.xlsx"
OUTPUT_PATH = "data/long_format_mixed_model.csv"

os.makedirs("data", exist_ok=True)


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format raw data to long format for mixed models.

    Input columns: 'Response ID', '<scenario> positive/negative[.N]', ...
    Output columns: participant_id, scenario, framing, rating
    """
    rows = []
    for _, row in df.iterrows():
        pid = int(row["Response ID"])
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
                        "scenario": scenario,
                        "framing": framing,
                        "rating": float(val),
                    }
                )

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(["participant_id", "scenario"]).reset_index(drop=True)
    # framing as dummy: positive=1, negative=0
    long_df["framing_dummy"] = (long_df["framing"] == "positive").astype(int)
    return long_df


def main():
    df = pd.read_excel(RAW_DATA_PATH)
    print(f"Loaded {len(df)} participants from {RAW_DATA_PATH}")

    long_df = wide_to_long(df)

    print(f"Long format shape: {long_df.shape}")
    print(f"Participants: {long_df['participant_id'].nunique()}")
    print(f"Scenarios: {sorted(long_df['scenario'].unique())}")
    print(f"Framing distribution:\n{long_df['framing'].value_counts()}")
    print(f"\nSample:\n{long_df.head(20).to_string()}")

    long_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
