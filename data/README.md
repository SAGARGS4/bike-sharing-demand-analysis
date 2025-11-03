# Dataset Information

## UCI Bike Sharing Dataset

This project uses the **Bike Sharing Dataset** from the UCI Machine Learning Repository.

### Download Instructions

1. Visit: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
2. Download the `day.csv` file
3. Place `day.csv` in this directory

### Dataset Description

The dataset contains daily bike rental counts along with weather and seasonal information.

**Features:**
- `instant`: Record index
- `dteday`: Date
- `season`: Season (1:spring, 2:summer, 3:fall, 4:winter)
- `yr`: Year (0: 2011, 1:2012)
- `mnth`: Month (1 to 12)
- `holiday`: Whether day is holiday or not
- `weekday`: Day of the week
- `workingday`: If day is neither weekend nor holiday
- `weathersit`: Weather situation
  - 1: Clear, Few clouds, Partly cloudy
  - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
  - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp`: Normalized temperature in Celsius
- `atemp`: Normalized feeling temperature in Celsius
- `hum`: Normalized humidity
- `windspeed`: Normalized wind speed
- `casual`: Count of casual users
- `registered`: Count of registered users
- `cnt`: Count of total rental bikes including both casual and registered

**Target Variable:** `cnt` (total bike rental count)
