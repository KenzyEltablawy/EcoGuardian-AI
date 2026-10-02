# Data

This folder contains the environmental dataset used by EcoGuardian AI.

## Files

- `environmental_dataset.csv` — City-level environmental data with temperature, humidity, wind speed, PM2.5, and PM10 measurements.

## Data Source

- World Health Organization (WHO) Ambient Air Quality Database
- Open-Meteo Weather and Air Quality APIs

## Preprocessing

1. Removed unnecessary columns
2. Standardized city and country names
3. Converted measurements to numerical values
4. Handled missing PM2.5 and PM10 values
5. Selected latest observation per city
