#------------------------------
# Step 1: Data Loading and Cleaning
#------------------------------
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("weather.csv", parse_dates=["Formatted Date"])

df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
df = df.set_index("Formatted Date")

df.dropna(inplace=True)

df["Summary"] = df["Summary"].str.lower()

def weather_type(summary):
    summary = summary.lower()  

    if "rain" in summary:
        return "rainy"
    elif "fog" in summary:
        return "foggy"
    elif "breeze" in summary or "wind" in summary:
        return "breezy"
    elif "overcast" in summary:
        return "overcast"
    elif "cloud" in summary:
        return "cloudy"
    elif "clear" in summary or "sunny" in summary:
        return "clear"
    else:
        return "other"

df["Weather Type"] = df["Summary"].apply(weather_type)

#------------------------------
# Step 2: Basic Statistics
#------------------------------
columns = ["Temperature (C)", "Humidity", "Wind Speed (km/h)"]
stats = df[columns].agg(['mean', 'median', 'max', 'min'])
print("\nBasic Statistics:")
print(stats)

weather_counts = df["Weather Type"].value_counts()
print("\nNumber of Days by Weather Type:")
print(weather_counts)

#------------------------------
# Step 3: Univariate Analysis
#------------------------------
plt.figure(figsize=(10, 5))
plt.hist(df["Temperature (C)"], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Average Temperatures")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.bar(weather_counts.index, weather_counts.values, color='lightcoral', edgecolor='black')
plt.title("Frequency of Weather Conditions")
plt.xlabel("Weather Type")
plt.ylabel("Number of Records")
plt.grid(axis='y')
plt.show()

#------------------------------
# Step 4: Time Series Trends
#------------------------------
plt.figure(figsize=(12, 5))
df["Temperature (C)"].resample("D").mean().plot(color='orange')
plt.title("Daily Average Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

daily_humidity = df["Humidity"].resample("D").mean()
daily_precip = df["Precip Type"].resample("D").count()

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel("Date")
ax1.set_ylabel("Humidity", color='tab:blue')
ax1.plot(daily_humidity, color='tab:blue', label="Humidity")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Precipitation Count", color='tab:green')
ax2.plot(daily_precip, color='tab:green', label="Precipitation")
ax2.tick_params(axis='y', labelcolor='tab:green')

plt.title("Daily Humidity and Precipitation Over Time")
fig.tight_layout()
plt.grid(True)
plt.show()

#------------------------------
# Step 5: Bivariate Analysis
#------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["Temperature (C)"], df["Humidity"], alpha=0.5, color='purple')
plt.title("Temperature vs. Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity")
plt.grid(True)
plt.tight_layout()
plt.show()

correlation = df[["Temperature (C)", "Humidity", "Wind Speed (km/h)"]].corr()
print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(6, 5))
plt.imshow(correlation, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df.boxplot(column="Temperature (C)", by="Weather Type", grid=False)
plt.title("Temperature by Weather Type")
plt.suptitle("")  
plt.xlabel("Weather Type")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

#------------------------------
# Step 6: Key Questions to Explore
#------------------------------
hottest_day = df["Temperature (C)"].idxmax()
coldest_day = df["Temperature (C)"].idxmin()

correlation_temp_humidity = df["Temperature (C)"].corr(df["Humidity"])

rainy = df[df["Weather Type"] == "rainy"]["Temperature (C)"]
non_rainy = df[df["Weather Type"] != "rainy"]["Temperature (C)"]

wind_by_weather = df.groupby("Weather Type")["Wind Speed (km/h)"].mean()

#------------------------------
# Step 8: Reporting 
#------------------------------
print("\n--- Summary Report ---")
print(f"Hottest Day: {hottest_day.date()} → {df.loc[hottest_day, 'Temperature (C)']:.2f}°C")
print(f"Coldest Day: {coldest_day.date()} → {df.loc[coldest_day, 'Temperature (C)']:.2f}°C")
print(f"\nTemperature-Humidity Correlation: {correlation_temp_humidity:.2f}")
print(f"\nAverage Temp on Rainy Days: {rainy.mean():.2f}°C")
print(f"Average Temp on Non-Rainy Days: {non_rainy.mean():.2f}°C")
print(f"\nAverage Wind Speeds by Weather Type:")
print(wind_by_weather)
