# Define season ranges
season_ranges = {
    "2019-20": range(24048, 24674),
    "2021-22": range(24785, 25465),
    "2022-23": range(25635, 26316),
    "2023-24": range(26478, 27158),
    "2024-25": list(range(27312, 27992)) # Convert to list so we can append missing game
}

# Add missing game to the corresponding season list
season_ranges["2024-25"].append(28042)

# Run the function to scrape shot data
df = scrape_ohl_data(season_ranges, max_workers=5, save_path="ohl_shots_2019_2025.parquet")