import pandas as pd

# Define file paths
path_unique_fighters = '../data/unique_fighters_with_country.csv'
path_general = '../data/UFC_PPV_Tableau.csv'

# --- Load DataFrames ---
try:
    df_unique_fighters = pd.read_csv(path_unique_fighters)
    print(f"Loaded '{path_unique_fighters}'. Shape: {df_unique_fighters.shape}")
    print("Columns:", df_unique_fighters.columns.tolist())
    print(df_unique_fighters.head())
    print("-" * 30)

    df_general = pd.read_csv(path_general)
    print(f"Loaded '{path_general}'. Shape: {df_general.shape}")
    print("Columns:", df_general.columns.tolist())
    print(df_general[['Opponent1', 'Opponent2']].head())
    print("-" * 30)

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both CSV files are in the specified '../data/' directory.")
    exit()

# --- Standardize Column Names and Clean Data for Comparison ---
# Ensure the 'Fighter' column exists in df_unique_fighters
if 'Fighter' not in df_unique_fighters.columns:
    print("Error: 'Fighter' column not found in unique_fighters_with_country.csv.")
    print("Please ensure your unique fighters CSV has a column named 'Fighter'.")
    exit()

# Convert unique fighter names to a set for faster lookup
# Apply stripping and title casing to ensure consistency with how Opponent1/2 will be cleaned
df_unique_fighters['Fighter Name Cleaned'] = df_unique_fighters['Fighter'].astype(str).str.strip().str.title().dropna()
unique_fighters_set = set(df_unique_fighters['Fighter Name Cleaned'].unique())
print(f"Total unique fighters in '{path_unique_fighters}': {len(unique_fighters_set)}")


# Clean 'Opponent1' and 'Opponent2' in df_general using the same logic
if 'Opponent1' in df_general.columns:
    df_general['Opponent1_Cleaned'] = df_general['Opponent1'].astype(str).str.strip().str.title()
else:
    print("Warning: 'Opponent1' column not found in df_general. Cannot check mismatches for this column.")
    df_general['Opponent1_Cleaned'] = pd.Series(dtype=str) # Create empty to avoid errors

if 'Opponent2' in df_general.columns:
    df_general['Opponent2_Cleaned'] = df_general['Opponent2'].astype(str).str.strip().str.title()
else:
    print("Warning: 'Opponent2' column not found in df_general. Cannot check mismatches for this column.")
    df_general['Opponent2_Cleaned'] = pd.Series(dtype=str) # Create empty to avoid errors


# --- Find Mismatched Fighters from df_general ---

# Check Opponent1
opponent1_not_in_unique_list = df_general[~df_general['Opponent1_Cleaned'].isin(unique_fighters_set)]
mismatched_opponent1_names = opponent1_not_in_unique_list['Opponent1_Cleaned'].unique().tolist()
mismatched_opponent1_names = [name for name in mismatched_opponent1_names if pd.notna(name) and name != 'Nan'] # Exclude actual NaN/string 'Nan' if any slipped through

# Check Opponent2
opponent2_not_in_unique_list = df_general[~df_general['Opponent2_Cleaned'].isin(unique_fighters_set)]
mismatched_opponent2_names = opponent2_not_in_unique_list['Opponent2_Cleaned'].unique().tolist()
mismatched_opponent2_names = [name for name in mismatched_opponent2_names if pd.notna(name) and name != 'Nan'] # Exclude actual NaN/string 'Nan' if any slipped through

# Combine and show unique mismatches
all_mismatched_fighters = sorted(list(set(mismatched_opponent1_names + mismatched_opponent2_names)))

# --- Report Findings ---
if all_mismatched_fighters:
    print("\n--- Mismatched Fighters Found! ---")
    print("These fighter names from UFC_PPV_Tableau.csv (after cleaning) are NOT in unique_fighters_with_country.csv:")
    for fighter in all_mismatched_fighters:
        print(f"- '{fighter}'")
    print(f"\nTotal unique mismatched fighters: {len(all_mismatched_fighters)}")


else:
    print("\n--- No Mismatched Fighters Found! ---")
    print("All cleaned fighter names from UFC_PPV_Tableau.csv appear to be present in unique_fighters_with_country.csv.")

# --- Additional Check: Fighters in unique_fighters_with_country.csv but not in df_general ---
# This checks for fighters in your unique list who never appeared in Opponent1 or Opponent2 in your general data.
# (This is less likely to be the cause of your Tableau NULLs, but good for data integrity)
all_fighters_in_general = set(df_general['Opponent1_Cleaned'].dropna().tolist() + df_general['Opponent2_Cleaned'].dropna().tolist())
fighters_in_unique_but_not_general = sorted(list(unique_fighters_set - all_fighters_in_general))

if fighters_in_unique_but_not_general:
    print("\n--- Fighters in unique_fighters_with_country.csv but NOT in UFC_PPV_Tableau.csv ---")
    print("These fighters are in your unique list but don't appear in Opponent1 or Opponent2 in your main data:")
    for fighter in fighters_in_unique_but_not_general:
        print(f"- '{fighter}'")
    print(f"\nTotal: {len(fighters_in_unique_but_not_general)}")
