import pandas as pd
import re

# --- Regex to match UFC Embedded ---
pattern = r"UFC (\d+) Embedded: Vlog Series - Episode (\d+)"

df = pd.read_csv('../data/ufc_video_data.csv')

# --- Step 1: Filter for 'UFC Embedded' titles and extract Event and Episode numbers ---
# We'll create temporary columns for easier processing
def extract_embedded_info(row_title):
    match = re.match(pattern, row_title)
    if match:
        event_num = int(match.group(1))
        episode_num = int(match.group(2))
        return pd.Series([f"UFC {event_num}", episode_num])
    return pd.Series([None, None]) # Return None for non-matching titles

# Apply the function to create temporary 'Extracted_Event' and 'Extracted_Episode' columns
df[['Extracted_Event', 'Extracted_Episode']] = df['Title'].apply(extract_embedded_info)

# Filter out rows that don't match the Embedded pattern
df_embedded = df.dropna(subset=['Extracted_Event', 'Extracted_Episode']).copy()

# Ensure Views is numeric for comparison
df_embedded['Views'] = pd.to_numeric(df_embedded['Views'], errors='coerce')
df_embedded.dropna(subset=['Views'], inplace=True) # Drop rows where Views couldn't be converted

# --- Step 2: Handle Duplicates - Keep the entry with the higher view count ---
# Sort by Event, Episode, and then by Views (descending)
# This way, when we drop duplicates, the highest view count for each (Event, Episode) will be kept.
df_embedded_sorted = df_embedded.sort_values(by=['Extracted_Event', 'Extracted_Episode', 'Views'], ascending=[True, True, False])

# Drop duplicates based on 'Extracted_Event' and 'Extracted_Episode', keeping the first (which is now the highest view count)
df_deduplicated = df_embedded_sorted.drop_duplicates(subset=['Extracted_Event', 'Extracted_Episode'], keep='first')

# --- Step 3: Pivot the data ---
# Now use the cleaned data for pivoting
df_pivot = df_deduplicated.pivot_table(
    index='Extracted_Event', # Use the cleaned Event number for index
    columns='Extracted_Episode', # Use the cleaned Episode number for columns
    values='Views',
    aggfunc='first' # 'first' is fine here since duplicates are handled
)

# Rename columns to 'Episode X' format
df_pivot.columns = [f'Episode {int(col)}' for col in df_pivot.columns]

# Fill any remaining NaN values (where an episode might be missing for a specific event) with 0
df_pivot = df_pivot.fillna(0)

# Reset index to make 'Extracted_Event' a regular column
df_pivot = df_pivot.reset_index()

# Rename the Event column for clarity
df_pivot = df_pivot.rename(columns={'Extracted_Event': 'Event'})

# --- Step 4: Save to CSV ---
df_pivot.to_csv('ufc_video_data_mergable_cleaned.csv', index=False)

print("\nProcessed and cleaned data saved to 'ufc_video_data_mergable_cleaned.csv'")
print("\nHead of the cleaned pivoted data:")
print(df_pivot.head())
print(f"\nShape of the cleaned pivoted data: {df_pivot.shape}")