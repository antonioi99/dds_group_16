import pandas as pd
import numpy as np
from typing import Dict, Any


def read_and_summarize_bike_data(file_path: str) -> Dict[str, Any]:
    """
    Read Seoul Bike Data CSV and return a comprehensive summary.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    dict : Dictionary containing dataset summary information
    """
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Create summary dictionary
    summary = {
        'basic_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d'),
                'unique_dates': df['Date'].nunique()
            },
            'missing_values': df.isnull().sum().sum()
        },
        
        'target_variable': {
            'name': 'Rented Bike Count',
            'mean': round(df['Rented Bike Count'].mean(), 2),
            'median': df['Rented Bike Count'].median(),
            'std': round(df['Rented Bike Count'].std(), 2),
            'min': df['Rented Bike Count'].min(),
            'max': df['Rented Bike Count'].max(),
            'zero_rentals': (df['Rented Bike Count'] == 0).sum()
        },
        
        'weather_features': {
            'temperature': {
                'mean': round(df['Temperature(°C)'].mean(), 2),
                'min': round(df['Temperature(°C)'].min(), 2),
                'max': round(df['Temperature(°C)'].max(), 2)
            },
            'humidity': {
                'mean': round(df['Humidity(%)'].mean(), 2),
                'min': df['Humidity(%)'].min(),
                'max': df['Humidity(%)'].max()
            },
            'wind_speed': {
                'mean': round(df['Wind speed (m/s)'].mean(), 2),
                'max': round(df['Wind speed (m/s)'].max(), 2)
            },
            'rainfall_days': (df['Rainfall(mm)'] > 0).sum(),
            'snowfall_days': (df['Snowfall (cm)'] > 0).sum()
        },
        
        'categorical_features': {
            'seasons': df['Seasons'].value_counts().to_dict(),
            'holidays': df['Holiday'].value_counts().to_dict(),
            'functioning_days': df['Functioning Day'].value_counts().to_dict()
        },
        
        'temporal_patterns': {
            'peak_hour': int(df.groupby('Hour')['Rented Bike Count'].mean().idxmax()),
            'lowest_hour': int(df.groupby('Hour')['Rented Bike Count'].mean().idxmin()),
            'avg_rentals_by_hour': df.groupby('Hour')['Rented Bike Count'].mean().round(2).to_dict()
        }
    }
    
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """
    Print the summary in a formatted way.
    
    Parameters:
    -----------
    summary : dict
        Summary dictionary from read_and_summarize_bike_data()
    """
    print("=" * 80)
    print("SEOUL BIKE DATA SUMMARY")
    print("=" * 80)
    
    # Basic Info
    print("\n📊 BASIC INFORMATION")
    print("-" * 80)
    basic = summary['basic_info']
    print(f"Total Records: {basic['total_rows']:,}")
    print(f"Total Features: {basic['total_columns']}")
    print(f"Date Range: {basic['date_range']['start']} to {basic['date_range']['end']}")
    print(f"Unique Dates: {basic['date_range']['unique_dates']}")
    print(f"Missing Values: {basic['missing_values']}")
    
    # Target Variable
    print("\n🎯 TARGET VARIABLE (Rented Bike Count)")
    print("-" * 80)
    target = summary['target_variable']
    print(f"Mean: {target['mean']:,.2f} bikes/hour")
    print(f"Median: {target['median']:,.0f} bikes/hour")
    print(f"Std Dev: {target['std']:,.2f}")
    print(f"Range: {target['min']} - {target['max']:,} bikes/hour")
    print(f"Hours with Zero Rentals: {target['zero_rentals']}")
    
    # Weather Features
    print("\n🌤️  WEATHER FEATURES")
    print("-" * 80)
    weather = summary['weather_features']
    print(f"Temperature: {weather['temperature']['mean']}°C (avg), "
          f"{weather['temperature']['min']}°C (min), "
          f"{weather['temperature']['max']}°C (max)")
    print(f"Humidity: {weather['humidity']['mean']}% (avg), "
          f"{weather['humidity']['min']}-{weather['humidity']['max']}% (range)")
    print(f"Wind Speed: {weather['wind_speed']['mean']} m/s (avg), "
          f"{weather['wind_speed']['max']} m/s (max)")
    print(f"Hours with Rainfall: {weather['rainfall_days']}")
    print(f"Hours with Snowfall: {weather['snowfall_days']}")
    
    # Categorical Features
    print("\n📅 CATEGORICAL FEATURES")
    print("-" * 80)
    categorical = summary['categorical_features']
    print("Seasons:")
    for season, count in categorical['seasons'].items():
        print(f"  {season}: {count:,} hours")
    print("\nHolidays:")
    for holiday, count in categorical['holidays'].items():
        print(f"  {holiday}: {count:,} hours")
    print("\nFunctioning Days:")
    for status, count in categorical['functioning_days'].items():
        print(f"  {status}: {count:,} hours")
    
    # Temporal Patterns
    print("\n⏰ TEMPORAL PATTERNS")
    print("-" * 80)
    temporal = summary['temporal_patterns']
    print(f"Peak Hour: {temporal['peak_hour']}:00 "
          f"(avg: {temporal['avg_rentals_by_hour'][temporal['peak_hour']]:,.2f} bikes)")
    print(f"Lowest Hour: {temporal['lowest_hour']}:00 "
          f"(avg: {temporal['avg_rentals_by_hour'][temporal['lowest_hour']]:,.2f} bikes)")
    
    print("\n" + "=" * 80)


# Example usage
if __name__ == "__main__":
    # Read and summarize the data
    file_path = "SeoulBikeData.csv"
    summary = read_and_summarize_bike_data(file_path)
    
    # Print formatted summary
    print_summary(summary)
    
    # You can also access individual summary components
    print("\n\nExample: Accessing specific summary data")
    print(f"Average rentals: {summary['target_variable']['mean']} bikes/hour")
    print(f"Peak rental hour: {summary['temporal_patterns']['peak_hour']}:00")