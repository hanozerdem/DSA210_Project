import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the datasets
weather_data = pd.read_csv(r"C:\Users\PC\Desktop\istanbul 2023-12-31 to 2024-12-31.csv")
viewing_data = pd.read_csv(r"C:\Users\PC\Desktop\ViewingActivity.csv")

# Preprocess weather data
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
weather_data['weather_category'] = weather_data['conditions']  # Keep the detailed conditions

# Find the minimum and maximum temperatures to define intervals
min_temp = int(np.floor(weather_data['temp'].min()))
max_temp = int(np.ceil(weather_data['temp'].max()))

# Create temperature intervals of size 5
bins = list(range(min_temp, max_temp + 5, 5))
labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins) - 1)]
weather_data['temp_interval'] = pd.cut(weather_data['temp'], bins=bins, labels=labels, right=False)

# Preprocess viewing data
viewing_data['Start Time'] = pd.to_datetime(viewing_data['Start Time'])
viewing_data['date'] = viewing_data['Start Time'].dt.date
viewing_data['duration_minutes'] = viewing_data['Duration'].apply(
    lambda x: sum(int(t) * 60**i for i, t in enumerate(reversed(x.split(":"))))
)

# Merge data based on date
weather_data['date'] = weather_data['datetime'].dt.date
merged_data = pd.merge(viewing_data, weather_data[['date', 'weather_category', 'temp_interval', 'temp']], on='date', how='inner')

# Visualization 1: Total viewing duration by weather condition (in hours)
detailed_watching_by_weather = merged_data.groupby('weather_category')['duration_minutes'].sum() / 60

# Normalize by the number of days for each weather condition (in hours)
days_per_condition = weather_data['weather_category'].value_counts()
normalized_weather = detailed_watching_by_weather / days_per_condition

# Plot total viewing duration by weather condition
plt.figure(figsize=(12, 8))
detailed_watching_by_weather.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Total Viewing Duration by Detailed Weather Conditions')
plt.ylabel('Total Viewing Duration (hours)')
plt.xlabel('Detailed Weather Conditions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot normalized viewing duration by weather condition
plt.figure(figsize=(12, 8))
normalized_weather.sort_values(ascending=False).plot(kind='bar', color='orange')
plt.title('Normalized Viewing Duration by Detailed Weather Conditions')
plt.ylabel('Average Viewing Duration per Day (hours)')
plt.xlabel('Detailed Weather Conditions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualization 2: Total viewing duration by temperature interval (in hours)
watching_by_temp = merged_data.groupby('temp_interval')['duration_minutes'].sum() / 60

# Normalize by the number of days for each temperature interval (in hours)
days_per_temp = weather_data['temp_interval'].value_counts()
normalized_temp = watching_by_temp / days_per_temp

# Plot total viewing duration by temperature intervals
plt.figure(figsize=(12, 8))
watching_by_temp.sort_index().plot(kind='bar', color='lightcoral')
plt.title('Total Viewing Duration by Temperature Intervals')
plt.ylabel('Total Viewing Duration (hours)')
plt.xlabel('Temperature Intervals (°C)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot normalized viewing duration by temperature intervals
plt.figure(figsize=(12, 8))
normalized_temp.sort_index().plot(kind='bar', color='green')
plt.title('Normalized Viewing Duration by Temperature Intervals')
plt.ylabel('Average Viewing Duration per Day (hours)')
plt.xlabel('Temperature Intervals (°C)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 1. Line Chart: Watching Habits Over Time vs. Weather
# Aggregate viewing data by day
daily_viewing = merged_data.groupby('date')['duration_minutes'].sum()

# Aggregate weather data by day (numeric only)
daily_weather = weather_data.groupby('date').mean(numeric_only=True)

# Convert viewing duration to hours
daily_viewing_hours = daily_viewing / 60

# Plot the line chart
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(daily_viewing_hours.index, daily_viewing_hours.values, label='Hours Watched', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('Hours Watched', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Add weather metrics on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(daily_weather.index, daily_weather['temp'], label='Average Temperature (°C)', color='red', alpha=0.6)
ax2.set_ylabel('Average Temperature (°C)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.tight_layout()
plt.title('Watching Habits Over Time vs. Weather')
plt.legend(loc='upper left')
plt.show()

# 2. Heatmap: Day of Week vs. Weather Impact
# Add day of the week to merged data
merged_data['day_of_week'] = pd.to_datetime(merged_data['date']).dt.day_name()

# Pivot table for average hours watched by day of week and weather category
heatmap_data = merged_data.pivot_table(
    values='duration_minutes',
    index='weather_category',
    columns='day_of_week',
    aggfunc='mean'
) / 60  # Convert minutes to hours

# Reorder columns to start with Monday
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data[days_order]
heatmap_data = heatmap_data.fillna(0)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Average Hours Watched'})
plt.title('Heatmap: Day of Week vs. Weather Impact')
plt.xlabel('Day of Week')
plt.ylabel('Weather Category')
plt.tight_layout()
plt.show()

# 3. Scatter Plot: Temperature vs. Watching Hours
# Aggregate viewing data by temperature
temp_vs_watching = merged_data.groupby('temp')['duration_minutes'].sum() / 60  # Convert minutes to hours

# Scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(temp_vs_watching.index, temp_vs_watching.values, alpha=0.6, color='purple', label='Watching Hours')
sns.regplot(x=temp_vs_watching.index, y=temp_vs_watching.values, scatter=False, color='red', label='Trendline')
plt.title('Scatter Plot: Temperature vs. Watching Hours')
plt.xlabel('Temperature (°C)')
plt.ylabel('Watching Hours')
plt.legend()
plt.tight_layout()
plt.show()
