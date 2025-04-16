import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
import os
import datetime
import joblib
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Oil Flow Forecasting Tool",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create necessary directories
models_dir = Path("models")
results_dir = Path("results")
models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# Custom CSS
st.markdown("""
<style>
.main {
    padding: 0rem 1rem;
}
.stApp {
    background-color: #f5f5f5;
}
.css-18e3th9 {
    padding-top: 1rem;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #2c3e50;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #f8f9fa;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 10px;
    padding-right: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #e6f2ff;
    border-bottom: 2px solid #1E88E5;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

# Load Data

def load_data():
    """Load all datasets with caching for streamlit."""
    try:
        supply_demand = pd.read_csv('SupplyDemand.2025-02-19T09-55.csv')
        regional_balance = pd.read_csv('Regional_Balance.2025-02-19T09-55.csv')
        maintenance = pd.read_csv('RefineryMaintenance.2025-02-19T09-55.csv')
        netback = pd.read_csv('Cleaned_Netback_Margin_Processed.csv')
        return supply_demand, regional_balance, maintenance, netback
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Make sure all CSV files are in the correct directory.")
        return None, None, None, None

# Get unique entities and flows
@st.cache_data(ttl=3600)
def get_unique_values(df, entity_col, flow_col):
    """Get unique entities and flows from the given dataframe."""
    entities = sorted(df[entity_col].unique())
    flows = sorted(df[flow_col].unique())
    return entities, flows

# Extract Target
def extract_target(supply_demand_df, regional_balance_df, entity_name, flow_breakdown, source_type):
    """Extracts the target time series based on entity, flow, and source type."""
    target_col_name = f"{entity_name}_{flow_breakdown}"
    
    if source_type == 'supply':
        source_df = supply_demand_df
        entity_col = 'CountryName'
    elif source_type == 'regional':
        source_df = regional_balance_df
        entity_col = 'GroupName'
    else:
        st.error("source_type must be 'supply' or 'regional'")
        return pd.DataFrame(), target_col_name

    target_data = source_df[
        (source_df[entity_col] == entity_name) &
        (source_df['FlowBreakdown'] == flow_breakdown)
    ].copy()

    if target_data.empty:
        st.warning(f"No data found for target {target_col_name}.")
        return pd.DataFrame(), target_col_name

    target_data['ReferenceDate'] = pd.to_datetime(target_data['ReferenceDate'])
    target_data = target_data.set_index('ReferenceDate')
    target_data = target_data[['ObservedValue']].rename(columns={'ObservedValue': target_col_name})
    target_data = target_data.sort_index()
    return target_data, target_col_name

# Generate entity features
def generate_entity_specific_features(supply_demand_df, regional_balance_df, entity_name, flow_breakdown_to_exclude, source_type):
    """Generates features from the target entity, excluding the target flow."""
    if source_type == 'supply':
        source_df = supply_demand_df
        entity_col = 'CountryName'
        prefix = f"{entity_name}_"
    elif source_type == 'regional':
        source_df = regional_balance_df
        entity_col = 'GroupName'
        prefix = f"{entity_name}_"
    else:
        return pd.DataFrame()

    entity_data = source_df[source_df[entity_col] == entity_name].copy()
    entity_data = entity_data[entity_data['FlowBreakdown'] != flow_breakdown_to_exclude]

    if entity_data.empty:
        st.info(f"No non-target features found for entity {entity_name}.")
        return pd.DataFrame()

    entity_data['ReferenceDate'] = pd.to_datetime(entity_data['ReferenceDate'])
    try:
        entity_features = entity_data.pivot_table(index='ReferenceDate', columns='FlowBreakdown', values='ObservedValue')
        entity_features = entity_features.add_prefix(prefix)
        return entity_features
    except Exception as e:
        st.warning(f"Error pivoting entity data for {entity_name}: {e}")
        # Handle potential duplicate entries
        entity_data_agg = entity_data.groupby(['ReferenceDate', 'FlowBreakdown'])['ObservedValue'].mean().reset_index()
        try:
             entity_features = entity_data_agg.pivot_table(index='ReferenceDate', columns='FlowBreakdown', values='ObservedValue')
             entity_features = entity_features.add_prefix(prefix)
             return entity_features
        except Exception as e2:
            st.error(f"Pivoting failed even after aggregation for {entity_name}: {e2}")
            return pd.DataFrame()

# Generate country features
def generate_country_features(supply_demand, countries, flow_breakdowns):
    """Generate features for selected countries."""
    country_features = pd.DataFrame()
    for country in countries:
        for flow in flow_breakdowns:
            country_data = supply_demand[
                (supply_demand['CountryName'] == country) &
                (supply_demand['FlowBreakdown'] == flow)
            ].copy()
            if not country_data.empty:
                country_data['ReferenceDate'] = pd.to_datetime(country_data['ReferenceDate'])
                country_data = country_data.set_index('ReferenceDate')
                col_name = f'{country}_{flow}'
                country_data = country_data[['ObservedValue']].rename(columns={'ObservedValue': col_name})
                if country_features.empty:
                    country_features = country_data
                else:
                    country_features = country_features.join(country_data, how='outer')

    # Aggregate potential duplicate indices
    if not country_features.empty:
        country_features = country_features.groupby(country_features.index).mean()
    return country_features

# Generate regional features
def generate_regional_features(regional_balance, regions, flow_breakdowns):
    """Generate features for selected regions."""
    regional_features = pd.DataFrame()
    for region in regions:
        for flow in flow_breakdowns:
            region_data = regional_balance[
                (regional_balance['GroupName'] == region) &
                (regional_balance['FlowBreakdown'] == flow)
            ].copy()
            if not region_data.empty:
                region_data['ReferenceDate'] = pd.to_datetime(region_data['ReferenceDate'])
                region_data = region_data.set_index('ReferenceDate')
                col_name = f'{region}_{flow}'
                region_data = region_data[['ObservedValue']].rename(columns={'ObservedValue': col_name})
                if regional_features.empty:
                    regional_features = region_data
                else:
                    regional_features = regional_features.join(region_data, how='outer')

    # Aggregate potential duplicate indices
    if not regional_features.empty:
        regional_features = regional_features.groupby(regional_features.index).mean()
    return regional_features

def generate_naive_forecast(y_train_series, forecast_index):
    """
    Generates a naive forecast (last value carried forward).

    Args:
        y_train_series (pd.Series): The historical target series used for training.
        forecast_index (pd.Index): The index (e.g., DatetimeIndex) for the forecast period.

    Returns:
        pd.Series: A series containing the naive forecast, aligned with forecast_index.
                   Returns empty series if training data is empty.
    """
    if y_train_series.empty:
        # Return an empty series with the correct index if train data is empty
        return pd.Series(index=forecast_index, dtype=float)
        
    last_value = y_train_series.iloc[-1]
    naive_forecast = pd.Series(last_value, index=forecast_index)
    return naive_forecast

def generate_seasonal_naive_forecast(y_train_series, forecast_index, season_length=12):
    """
    Generates a seasonal naive forecast (using value from same season last cycle).
    
    Args:
        y_train_series (pd.Series): The historical target series used for training.
        forecast_index (pd.Index): The index (e.g., DatetimeIndex) for the forecast period.
        season_length (int): The seasonality period (default=12 for monthly data).
        
    Returns:
        pd.Series: A series containing the seasonal naive forecast, aligned with forecast_index.
                  Returns empty series if training data is empty or shorter than season_length.
    """
    if y_train_series.empty or len(y_train_series) < season_length:
        # Return an empty series with the correct index if train data is insufficient
        return pd.Series(index=forecast_index, dtype=float)
    
    # Create a forecast series with the correct index
    seasonal_forecast = pd.Series(index=forecast_index, dtype=float)
    
    # For each forecast point, find the value from season_length steps ago
    for forecast_date in forecast_index:
        # Get same season from previous cycle
        # For monthly data with season_length=12, this gets the same month from previous year
        season_offset = pd.DateOffset(months=season_length)
        historical_date = forecast_date - season_offset
        
        # If historical date exists in training data, use that value
        if historical_date in y_train_series.index:
            seasonal_forecast.loc[forecast_date] = y_train_series.loc[historical_date]
        else:
            # Find the closest available date in training data
            closest_dates = [date for date in y_train_series.index 
                           if abs((date.month - historical_date.month)) <= 1]
            if closest_dates:
                # Use the most recent similar seasonal point
                closest_date = max([d for d in closest_dates if d <= forecast_date])
                seasonal_forecast.loc[forecast_date] = y_train_series.loc[closest_date]
            else:
                # Fallback to last value if no good seasonal match
                seasonal_forecast.loc[forecast_date] = y_train_series.iloc[-1]
    
    return seasonal_forecast

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def optimize_xgboost_hyperparameters(X_train, y_train, X_val=None, y_val=None, cv=3, n_iter=10):
    """
    Performs simple hyperparameter optimization for XGBoost using RandomizedSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame, optional): Validation features. If provided, will use val set instead of CV.
        y_val (pd.Series, optional): Validation target.
        cv (int): Number of cross-validation folds if val set not provided.
        n_iter (int): Number of parameter settings sampled.
        
    Returns:
        dict: Best parameters found during optimization.
    """
    # Define the hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5]
    }
    
    # Create base model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=20,
        eval_metric='rmse'
    )
    
    # Handle NaNs and Infs in the data
    X_train_clean = X_train.copy()
    X_train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute remaining NaNs
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_clean),
        columns=X_train_clean.columns,
        index=X_train_clean.index
    )
    
    # Clean y_train
    y_train_clean = y_train.copy()
    if y_train_clean.isnull().any():
        valid_indices = y_train_clean.dropna().index
        y_train_clean = y_train_clean.loc[valid_indices]
        X_train_imputed = X_train_imputed.loc[valid_indices]
    
    # If validation set provided, use it instead of CV
    if X_val is not None and y_val is not None:
        # Clean validation data
        X_val_clean = X_val.copy()
        X_val_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_val_imputed = pd.DataFrame(
            imputer.transform(X_val_clean),
            columns=X_val_clean.columns,
            index=X_val_clean.index
        )
        
        y_val_clean = y_val.copy()
        if y_val_clean.isnull().any():
            valid_val_indices = y_val_clean.dropna().index
            y_val_clean = y_val_clean.loc[valid_val_indices]
            X_val_imputed = X_val_imputed.loc[valid_val_indices]
        
        # Create fit params for early stopping with validation set
        fit_params = {
            'eval_set': [(X_val_imputed, y_val_clean)],
            'verbose': False
        }
        
        # Use 1-fold CV since we have a separate validation set
        search = RandomizedSearchCV(
            model, param_distributions=param_dist, 
            n_iter=n_iter, scoring='neg_mean_squared_error',
            cv=[(np.arange(len(X_train_imputed)), [])],  # Dummy CV
            n_jobs=-1, random_state=42, refit=False
        )
        
        # Fit with early stopping using validation set
        search.fit(X_train_imputed, y_train_clean, **fit_params)
    else:
        # Use standard CV approach
        search = RandomizedSearchCV(
            model, param_distributions=param_dist, 
            n_iter=n_iter, scoring='neg_mean_squared_error',
            cv=cv, n_jobs=-1, random_state=42
        )
        
        # Fit using cross-validation
        search.fit(X_train_imputed, y_train_clean)
    
    # Return the best parameters
    return search.best_params_

def generate_maintenance_features(maintenance_df, countries_list):
    """
    Generate maintenance features for a list of selected countries.
    Each country will have its own feature columns (e.g., USA_Capacity_Offline).

    Args:
        maintenance_df (pd.DataFrame): The raw refinery maintenance dataframe.
        countries_list (list): A list of country names selected by the user.

    Returns:
        pd.DataFrame: A dataframe indexed by monthly start dates, containing
                      Maintenance_Count and Capacity_Offline columns for each
                      selected country. Returns empty DataFrame if no countries
                      selected or no data found.
    """
    if not countries_list:
        # st.info("No maintenance countries selected.") # Keep UI cleaner
        return pd.DataFrame()

    # Filter first to relevant countries to avoid unnecessary processing
    relevant_maintenance = maintenance_df[maintenance_df['Country'].isin(countries_list)].copy()
    if relevant_maintenance.empty:
        st.warning(f"No maintenance data found for selected countries: {', '.join(countries_list)}.")
        return pd.DataFrame()

    # --- Robust Date Parsing & Cleaning ---
    # Parse dates using dayfirst=True based on observed DD/MM/YYYY format
    relevant_maintenance['StartDate'] = pd.to_datetime(relevant_maintenance['StartDate'], errors='coerce', dayfirst=True)
    relevant_maintenance['EndDate'] = pd.to_datetime(relevant_maintenance['EndDate'], errors='coerce', dayfirst=True)
    # Clean capacity - ensure numeric, fill NaNs resulting from coercion with 0
    relevant_maintenance['CapacityOffline'] = pd.to_numeric(relevant_maintenance['CapacityOffline'], errors='coerce').fillna(0)

    # Drop rows where essential dates couldn't be parsed
    relevant_maintenance.dropna(subset=['StartDate', 'EndDate'], inplace=True)

    if relevant_maintenance.empty:
        st.warning(f"No valid maintenance date entries found for selected countries after parsing.")
        return pd.DataFrame()
    # --- End Date Parsing & Cleaning ---

    # --- Determine Consistent Date Range ---
    # Find the absolute min/max dates across all relevant events for consistent indexing
    min_date_overall = relevant_maintenance['StartDate'].min()
    max_date_overall = relevant_maintenance['EndDate'].max() # Use max end date for range

    # Use current date if max_date is invalid or only covers the past
    today = pd.Timestamp.now().normalize()
    # Handle case where max_date_overall might be NaT after filtering/dropna
    if pd.isna(max_date_overall) or max_date_overall < today:
         # If no end dates, still use today as the max bound for index generation
         max_date_overall = today
    # If min_date is NaT (shouldn't happen if relevant_maintenance is not empty after dropna, but safer)
    if pd.isna(min_date_overall):
         st.warning(f"Could not determine a valid start date for maintenance features.")
         return pd.DataFrame()

    # Create the full monthly date index covering the entire span
    # Ensure start date is beginning of month, end date covers the last month needed
    try:
        monthly_dates_index = pd.date_range(
            start=min_date_overall.replace(day=1),
            # Go to the end of the month containing the max relevant end date
            end=max_date_overall + pd.offsets.MonthEnd(0),
            freq='MS' # Monthly Start frequency
        )
    except ValueError as e:
        st.error(f"Error creating date range for maintenance features: {e}. Min Date: {min_date_overall}, Max Date: {max_date_overall}")
        return pd.DataFrame()
    # --- End Determine Consistent Date Range ---


    # --- Generate Features Per Country ---
    all_maintenance_features = pd.DataFrame(index=monthly_dates_index) # Initialize with the common index

    # Loop through each selected country to generate its specific features
    for country in countries_list:
        # Filter the already cleaned relevant_maintenance data for the current country
        country_maintenance = relevant_maintenance[relevant_maintenance['Country'] == country] # No need for .copy() here

        if country_maintenance.empty:
            # Optionally log or inform, but don't stop processing other countries
            # print(f"Debug: No maintenance data for {country} within the cleaned set.")
            continue # Skip to next country

        # Create temporary Series to store this country's calculated values, aligned to the common index
        country_maint_count = pd.Series(0, index=monthly_dates_index, dtype=int)
        country_cap_offline = pd.Series(0.0, index=monthly_dates_index, dtype=float)

        # Iterate through the common monthly dates to calculate features
        for date in monthly_dates_index:
            month_start = date
            month_end = date + pd.offsets.MonthEnd(0)

            # Find events active during this month FOR THIS COUNTRY
            # Compare against the already parsed datetime columns
            active_events = country_maintenance[
                ((country_maintenance['StartDate'] <= month_end) &
                 (country_maintenance['EndDate'] >= month_start))
            ]

            # Assign calculated values for this month to the temporary Series
            country_maint_count.loc[date] = len(active_events)
            country_cap_offline.loc[date] = active_events['CapacityOffline'].sum() # CapacityOffline is already cleaned numeric

        # Add this country's calculated features as new columns to the main DataFrame
        # Use country name in column name for clarity (Option C)
        all_maintenance_features[f'{country}_Maintenance_Count'] = country_maint_count
        all_maintenance_features[f'{country}_Capacity_Offline'] = country_cap_offline

    # The resulting DataFrame 'all_maintenance_features' now has columns for each selected country
    # No need for further joins as we built it column by column on the common index
    return all_maintenance_features

# ... (other functions like generate_market_features, generate_trend_features etc. remain unchanged) ...
# ... (other functions remain the same) ...
# Generate market features
def generate_market_features(netback):
    """Generate market features from netback data."""
    netback = netback.copy()
    netback['Dates'] = pd.to_datetime(netback['Dates'])
    # Resample to monthly frequency
    numeric_cols = netback.select_dtypes(include=np.number).columns
    monthly_netback = netback.set_index('Dates')[numeric_cols].resample('MS').mean()

    # Calculate cracks and spreads
    if '92 Ron Gasoline' in monthly_netback.columns and 'Dubai Crude' in monthly_netback.columns:
        monthly_netback['Gasoline_Crack'] = monthly_netback['92 Ron Gasoline'] - monthly_netback['Dubai Crude']
    if 'Diesel 0.5 ppm' in monthly_netback.columns and 'Dubai Crude' in monthly_netback.columns:
        monthly_netback['Diesel_Crack'] = monthly_netback['Diesel 0.5 ppm'] - monthly_netback['Dubai Crude']
    if 'Jet Fuel' in monthly_netback.columns and 'Dubai Crude' in monthly_netback.columns:
        monthly_netback['Jet_Crack'] = monthly_netback['Jet Fuel'] - monthly_netback['Dubai Crude']

    # Calculate spreads
    if '92 Ron Gasoline' in monthly_netback.columns and 'Diesel 0.5 ppm' in monthly_netback.columns:
        monthly_netback['Gasoline_Diesel_Spread'] = monthly_netback['92 Ron Gasoline'] - monthly_netback['Diesel 0.5 ppm']
    if '92 Ron Gasoline' in monthly_netback.columns and 'Jet Fuel' in monthly_netback.columns:
        monthly_netback['Gasoline_Jet_Spread'] = monthly_netback['92 Ron Gasoline'] - monthly_netback['Jet Fuel']

    # Add volatility metrics (standard deviation over 3-month window)
    if 'Dubai Crude' in monthly_netback.columns:
        monthly_netback['Dubai_Crude_Volatility'] = monthly_netback['Dubai Crude'].rolling(window=3).std()

    return monthly_netback

# Generate trend features
def generate_trend_features(df):
    """Add explicit trend features."""
    df = df.copy()
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        st.warning("Cannot generate trend features - empty or non-DatetimeIndex DataFrame.")
        return df

    earliest_date = df.index.min()
    if pd.isna(earliest_date):
        st.warning("Cannot determine earliest date for trend features.")
        return df

    df['months_since_start'] = ((df.index.year - earliest_date.year) * 12 +
                              (df.index.month - earliest_date.month))
    df['years_since_start'] = df['months_since_start'] / 12
    df['trend_squared'] = df['months_since_start'] ** 2
    df['trend_cubed'] = df['months_since_start'] ** 3
    
    # Add cyclical encoding of month (better than one-hot)
    df['month_sin'] = np.sin(df.index.month * (2 * np.pi / 12))
    df['month_cos'] = np.cos(df.index.month * (2 * np.pi / 12))
    
    # Add quarter cyclical encoding
    df['quarter_sin'] = np.sin(df.index.quarter * (2 * np.pi / 4))
    df['quarter_cos'] = np.cos(df.index.quarter * (2 * np.pi / 4))
    
    return df

# Generate time features
def generate_time_features(df):
    """Add time features."""
    df = df.copy()
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        st.warning("Cannot generate time features - empty or non-DatetimeIndex DataFrame.")
        return df

    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year
    return df

# Generate derived features
def generate_derived_features(df, target_col_name, lags=[1, 3, 6, 12], windows=[3, 6, 12]):
    """Generates lags, moving averages, and YoY change."""
    df = df.copy()
    if target_col_name not in df.columns:
        st.warning(f"Target column '{target_col_name}' not found for generating derived features.")
        return df

    # Lagged features for target
    for lag in lags:
        df[f'{target_col_name}_lag_{lag}'] = df[target_col_name].shift(lag)

    # Moving averages for target
    for window in windows:
        df[f'{target_col_name}_ma_{window}'] = df[target_col_name].rolling(window=window, min_periods=1).mean()

    # Year-over-year change
    df[f'{target_col_name}_yoy'] = df[target_col_name].pct_change(periods=12)
    # Replace potential inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Create lagged features for other potentially important variables
    important_feature_keywords = ['REFINOBS', 'REFCAP', 'CLOSTLV', 'DIRECUSE', 'TOTIMPSB', 
                                 'Dubai_Netback_Margin', 'Crack', 'Crude', 'Capacity_Offline']
    cols_to_lag = [
        col for col in df.columns
        if any(keyword in col for keyword in important_feature_keywords)
        and col != target_col_name
        and '_lag_' not in col
        and '_ma_' not in col
        and '_yoy' not in col
    ]

    # Add basic lags for selected features
    for col in cols_to_lag:
        for lag in [1, 3]:
             df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Create moving averages for market variables
    market_cols = [col for col in df.columns if any(x in col for x in ['Margin', 'Crack', 'Crude', 'Spread']) 
                  and '_lag_' not in col and '_ma_' not in col]
    for col in market_cols:
        for window in [3, 6]:
             df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()

    return df

# Combine data
def combine_data(target_df, feature_dfs):
    """Combines target with feature dataframes."""
    if target_df.empty:
        st.error("Target DataFrame is empty, cannot combine.")
        return pd.DataFrame()

    # Start with target dataframe
    combined_df = target_df.copy()

    # Join all feature dataframes
    for i, feature_df in enumerate(feature_dfs):
        if feature_df is not None and not feature_df.empty:
            if not isinstance(feature_df.index, pd.DatetimeIndex):
                try:
                    feature_df.index = pd.to_datetime(feature_df.index)
                except Exception as e:
                    st.warning(f"Error converting index for feature set {i+1}: {e}. Skipping.")
                    continue

            # Ensure feature index is sorted
            feature_df = feature_df.sort_index()

            # Join with outer join to keep all dates
            combined_df = combined_df.join(feature_df, how='outer')
             
            # Check for duplicate columns
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]

    # Ensure index is unique and sorted
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()

    # Fill NaNs: forward fill first, then backfill remaining
    combined_df = combined_df.ffill().bfill()

    # Re-align to original target index dates
    combined_df = combined_df.loc[combined_df.index.isin(target_df.index)]
    
    # Drop rows where target is NaN
    target_col = target_df.columns[0]
    combined_df.dropna(subset=[target_col], inplace=True)

    return combined_df

# Feature selection with RFE
def select_features_with_rfe(X_train, y_train, n_features_to_select=30, target_name="Target"):
    """Performs Recursive Feature Elimination to select the most important features."""
    if X_train.empty:
        st.error("Training data is empty, cannot perform feature selection.")
        return [], pd.DataFrame()

    # Handle NaNs and Infs
    X_train_clean = X_train.copy()
    X_train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute remaining NaNs
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_clean),
        columns=X_train_clean.columns,
        index=X_train_clean.index
    )

    # Clean y_train
    y_train_clean = y_train.copy()
    if y_train_clean.isnull().any():
        valid_indices = y_train_clean.dropna().index
        y_train_clean = y_train_clean.loc[valid_indices]
        X_train_imputed = X_train_imputed.loc[valid_indices]

    if X_train_imputed.empty or y_train_clean.empty:
        st.error("Data is empty after cleaning, cannot perform feature selection.")
        return [], pd.DataFrame()

    # Adjust n_features_to_select if needed
    actual_n_features = X_train_imputed.shape[1]
    n_features_to_select = min(max(1, n_features_to_select), actual_n_features)

    # Create base model for RFE
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Run RFE
    rfe = RFE(
        estimator=model,
        n_features_to_select=n_features_to_select,
        step=0.1
    )

    try:
        rfe.fit(X_train_imputed, y_train_clean)
    except Exception as e:
        st.error(f"Error during feature selection: {e}")
        # Fallback to simple feature importance
        try:
            model.fit(X_train_imputed, y_train_clean)
            importance_df = pd.DataFrame({
                'Feature': X_train_imputed.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            selected_features = importance_df.head(n_features_to_select)['Feature'].tolist()
            feature_ranking = importance_df.reset_index(drop=True)
            feature_ranking['Rank'] = feature_ranking.index + 1
            return selected_features, feature_ranking
        except Exception:
            st.error("Feature selection failed completely.")
            return [], pd.DataFrame()

    # Get selected features and ranking
    selected_features = X_train_imputed.columns[rfe.support_].tolist()
    feature_ranking = pd.DataFrame({
        'Feature': X_train_imputed.columns,
        'Rank': rfe.ranking_
    }).sort_values('Rank')

    return selected_features, feature_ranking

# Train and evaluate model
def train_and_evaluate_model(X_train, y_train, X_test, y_test, selected_features, target_name="Target", use_hpo=False):
    """Trains and evaluates XGBoost model using selected features."""
    if not selected_features:
        st.error("No features selected for modeling.")
        return None, None, {}, {}

    if X_train.empty or X_test.empty:
        st.error("Training or testing data is empty.")
        return None, None, {}, {}

    # Filter to selected features
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()

    # Handle NaNs/Infs
    X_train_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_selected.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_selected),
        columns=X_train_selected.columns,
        index=X_train_selected.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test_selected),
        columns=X_test_selected.columns,
        index=X_test_selected.index
    )

    # Clean target variables
    y_train_clean = y_train.copy()
    y_test_clean = y_test.copy()

    if y_train_clean.isnull().any():
        valid_train_indices = y_train_clean.dropna().index
        y_train_clean = y_train_clean.loc[valid_train_indices]
        X_train_imputed = X_train_imputed.loc[valid_train_indices]

    if y_test_clean.isnull().any():
        valid_test_indices = y_test_clean.dropna().index
        y_test_clean = y_test_clean.loc[valid_test_indices]
        X_test_imputed = X_test_imputed.loc[valid_test_indices]

    if X_train_imputed.empty or y_train_clean.empty or X_test_imputed.empty or y_test_clean.empty:
        st.error("Data became empty after cleaning.")
        return None, None, {}, {}

    # Get model parameters (optimized or default)
    if use_hpo:
        with st.spinner(f"Optimizing hyperparameters for {target_name}..."):
            # Create small validation set from training data
            # Use last 20% of training data as validation set
            val_size = int(0.2 * len(X_train_imputed))
            X_val = X_train_imputed.iloc[-val_size:]
            y_val = y_train_clean.iloc[-val_size:]
            X_train_hpo = X_train_imputed.iloc[:-val_size]
            y_train_hpo = y_train_clean.iloc[:-val_size]
            
            best_params = optimize_xgboost_hyperparameters(
                X_train_hpo, y_train_hpo, X_val, y_val, n_iter=10
            )
            
            # Initialize model with optimized parameters
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                eval_metric='rmse',
                early_stopping_rounds=50,
                **best_params
            )
            
            st.success("Hyperparameter optimization completed!")
    else:
        # Initialize XGBoost model with default parameters
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=1,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=50
        )

    # Train model
    with st.spinner(f"Training model for {target_name}..."):
        model.fit(
            X_train_imputed, y_train_clean,
            eval_set=[(X_test_imputed, y_test_clean)],
            verbose=False
        )

    # Make predictions
    y_pred = model.predict(X_test_imputed)
    y_pred_series = pd.Series(y_pred, index=X_test_imputed.index)
    
    # Make training predictions for backtesting
    y_train_pred = model.predict(X_train_imputed)
    y_train_pred_series = pd.Series(y_train_pred, index=X_train_imputed.index)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred))
    mae = mean_absolute_error(y_test_clean, y_pred)
    r2 = r2_score(y_test_clean, y_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test_clean - y_pred) / y_test_clean)) * 100
    
    # Directional accuracy
    actual_direction = np.sign(y_test_clean.diff().dropna())
    pred_direction = np.sign(y_pred_series.diff().dropna())
    # Align indices
    common_indices = actual_direction.index.intersection(pred_direction.index)
    directional_accuracy = np.mean(actual_direction[common_indices] == pred_direction[common_indices]) * 100

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape,
        'Directional Accuracy (%)': directional_accuracy,
        'Best Iteration': model.best_iteration
    }

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train_imputed.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Create predictions DataFrame with both training and test predictions
    all_predictions = pd.DataFrame({
        'Actual': pd.concat([y_train_clean, y_test_clean]),
        'Predicted': pd.concat([y_train_pred_series, y_pred_series]),
        'Set': ['Train'] * len(y_train_clean) + ['Test'] * len(y_test_clean)
    })
    all_predictions.index.name = 'Date'
    
    return model, all_predictions, metrics, feature_importance
# --- UI Components ---

def show_sidebar():
    """Configure sidebar for app settings."""
    st.sidebar.title("⚙️ Settings")
    
    # Data loading status
    with st.sidebar.expander("📊 Data Status", expanded=False):
        if 'data_loaded' in st.session_state and st.session_state.data_loaded:
            st.success("All datasets loaded successfully!")
            
            # Show data stats
            st.write("**Supply Demand Data:**")
            st.write(f"- Countries: {len(st.session_state.countries)}")
            st.write(f"- Flows: {len(st.session_state.supply_flows)}")
            
            st.write("**Regional Balance Data:**")
            st.write(f"- Regions: {len(st.session_state.regions)}")
            st.write(f"- Flows: {len(st.session_state.regional_flows)}")
        else:
            st.warning("Data not loaded. Please load data first.")
    
    # Model settings
    with st.sidebar.expander("🧠 Model Settings", expanded=True):
        n_features = st.slider("Number of features to select", 10, 50, 30)
        test_size = st.slider("Test set size (%)", 10, 50, 20)
        
        # Store in session state
        st.session_state.n_features = n_features
        st.session_state.test_size = test_size / 100
    
    # Feature generation settings
    with st.sidebar.expander("🧩 Feature Engineering", expanded=False):
        lag_options = st.multiselect(
            "Lag periods (months)",
            options=[1, 2, 3, 6, 12, 24],
            default=[1, 3, 6, 12]
        )
        
        ma_options = st.multiselect(
            "Moving average windows (months)",
            options=[3, 6, 12, 24],
            default=[3, 6, 12]
        )
        
        st.session_state.lag_options = lag_options
        st.session_state.ma_options = ma_options
    
    # About section
    with st.sidebar.expander("ℹ️ About", expanded=False):
        st.write("""
        **Oil Flow Forecasting Tool**
        
        This application forecasts oil production, consumption, and trade flows 
        for countries and regions worldwide.
        
        *Built with Streamlit and XGBoost*
        """)
        
        st.write("---")
        st.write(f"Last updated: {datetime.date.today().strftime('%B %d, %Y')}")

#@st.cache_resource(ttl=3600)
def main_forecasting(supply_demand, regional_balance, maintenance, netback):
    """Main forecasting page functionality."""
    st.title("🛢️ Oil Flow Forecasting")
    
    # Entity selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        source_type = st.radio("Select Source Type:", 
                              ["Country (Supply Demand)", "Region (Regional Balance)"], 
                              index=0,
                              help="Choose between country-level or regional-level data")
    
    # Determine entity type and options based on selection
    if source_type == "Country (Supply Demand)":
        entity_type = "supply"
        entity_label = "Country"
        entity_options = st.session_state.countries
        flow_options = st.session_state.supply_flows
    else:
        entity_type = "regional"
        entity_label = "Region"
        entity_options = st.session_state.regions
        flow_options = st.session_state.regional_flows
    
    with col2:
        selected_entity = st.selectbox(f"Select {entity_label}:", entity_options)
    
    with col3:
        selected_flow = st.selectbox("Select Flow:", flow_options)
    
    st.divider()
    
    # Check selected target exists
    with st.spinner(f"Checking availability of {selected_entity} {selected_flow} data..."):
        if entity_type == "supply":
            has_data = not supply_demand[(supply_demand['CountryName'] == selected_entity) & 
                                      (supply_demand['FlowBreakdown'] == selected_flow)].empty
        else:
            has_data = not regional_balance[(regional_balance['GroupName'] == selected_entity) & 
                                         (regional_balance['FlowBreakdown'] == selected_flow)].empty
    
    if not has_data:
        st.error(f"No data available for {selected_entity} {selected_flow}. Please select a different combination.")
        return
    
    # Forecast settings
    st.subheader("Forecast Settings")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4) # Use 4 columns for better layout

    with col_f1:
        # Select related countries for features (SupplyDemand data)
        default_countries = ['China', 'United States', 'Russia', 'Saudi Arabia']
        default_countries = [c for c in default_countries if c in st.session_state.countries and c != selected_entity]
        related_countries = st.multiselect(
            "Related Countries for Features:",
            options=[c for c in st.session_state.countries if c != selected_entity],
            default=default_countries[:3],
            key="fc_related_countries" # Add unique key
        )

    with col_f2:
        # Select related regions for features (RegionalBalance data)
        default_regions = ['OECD Asia Oceania', 'Other Asia', 'China', 'Africa']
        default_regions = [r for r in default_regions if r in st.session_state.regions and r != selected_entity]
        related_regions = st.multiselect(
            "Related Regions for Features:",
            options=[r for r in st.session_state.regions if r != selected_entity],
            default=default_regions[:2],
            key="fc_related_regions" # Add unique key
        )

    with col_f3:
        # Select key flows for related entities
        default_flows = ['REFINOBS', 'REFCAP', 'CLOSTLV', 'TOTIMPSB']
        default_flows = [f for f in default_flows if f in flow_options and f != selected_flow]
        related_flows = st.multiselect(
            "Related Flows for Features:",
            options=[f for f in flow_options if f != selected_flow],
            default=default_flows[:3],
            key="fc_related_flows" # Add unique key
        )

    with col_f4:
         # Select countries for Maintenance features
         available_maint_countries = []
         # Check if maintenance data exists and is not empty before accessing .unique()
         if 'maintenance' in st.session_state and isinstance(st.session_state.maintenance, pd.DataFrame) and not st.session_state.maintenance.empty:
              try:
                   available_maint_countries = sorted(st.session_state.maintenance['Country'].unique())
              except KeyError:
                   st.warning("Maintenance data loaded but missing 'Country' column.")
                   available_maint_countries = []
         else:
              st.warning("Maintenance data not loaded or empty.")

         # --- Determine Defaults ---
         default_maint_countries = []
         # 1. Add target country if applicable and available
         if entity_type == 'supply' and selected_entity in available_maint_countries:
             default_maint_countries.append(selected_entity)

         # 2. Add 1-2 major hubs if available and not already added (customize list as needed)
         major_hubs_to_consider = ['United States', 'China'] # Example hubs
         for hub in major_hubs_to_consider:
             if hub in available_maint_countries and hub not in default_maint_countries:
                 # Limit the number of default hubs if desired (e.g., add only the first found)
                 if len(default_maint_countries) < 3: # Example: Limit total defaults to max 3 (target + 2 hubs)
                      default_maint_countries.append(hub)

         # If still empty after checks (e.g., regional target, no hubs found), default remains []
         # --- End Determine Defaults ---

         selected_maintenance_countries = st.multiselect(
             "Countries for Maintenance Features:",
             options=available_maint_countries,
             default=default_maint_countries, # Use the dynamically determined list
             key="fc_maint_countries"
         )
    # --- END NEW WIDGET ---

    # Run forecast button
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Extracting target variable..."):
            # Extract target
            target_df, target_col_name = extract_target(
                supply_demand, regional_balance, selected_entity, selected_flow, entity_type
            )
            
            if target_df.empty:
                st.error(f"Could not extract target data for {selected_entity} {selected_flow}.")
                return
            
            # Display target preview
            st.subheader(f"Target Variable: {target_col_name}")
            
            # Plot target time series
            fig = px.line(
                target_df, 
                x=target_df.index, 
                y=target_col_name,
                title=f"{target_col_name} Historical Data",
                labels={"value": "Value", "ReferenceDate": "Date"},
                template="plotly_white"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Generate features
        with st.spinner("Generating features..."):
            progress_bar = st.progress(0)
            
            # Entity-specific features (excluding target flow)
            progress_bar.progress(10)
            entity_features = generate_entity_specific_features(
                supply_demand, regional_balance, selected_entity, selected_flow, entity_type
            )
            
            # Related country features
            progress_bar.progress(25)
            country_features = generate_country_features(
                supply_demand, related_countries, related_flows
            )
            
            # Related region features
            progress_bar.progress(40)
            region_features = generate_regional_features(
                regional_balance, related_regions, related_flows
            )
            
            # --- MODIFIED CALL TO generate_maintenance_features ---
            progress_bar.progress(60)
            # Pass the full maintenance DataFrame and the list of selected countries
            # The function now handles iterating through the list and returns features
            # with country-specific column names (Option C)
            maintenance_features = generate_maintenance_features(
                st.session_state.maintenance,
                selected_maintenance_countries
            )
            
            # Market features
            progress_bar.progress(80)
            market_features = generate_market_features(netback)
            
            # Combine all feature sets
            progress_bar.progress(90)
            feature_dfs = [
                entity_features, country_features, region_features,
                maintenance_features, market_features
            ]
            # Filter out empty DataFrames
            feature_dfs = [df for df in feature_dfs if df is not None and not df.empty]
            
            # Combine target with features
            combined_df = combine_data(target_df, feature_dfs)
            progress_bar.progress(100)
            
            if combined_df.empty:
                st.error("Failed to combine features with target data.")
                return
            
            st.success(f"Generated {combined_df.shape[1]-1} initial features!")
        
        # Generate derived features
        with st.spinner("Engineering time-based and derived features..."):
            # Add time features
            combined_df = generate_time_features(combined_df)
            
            # Add trend features
            combined_df = generate_trend_features(combined_df)
            
            # Add derived features (lags, MAs, etc.)
            final_df = generate_derived_features(
                combined_df, target_col_name, 
                lags=st.session_state.lag_options, 
                windows=st.session_state.ma_options
            )
            
            # Prepare X and y
            final_df = final_df.sort_index()
            final_df.dropna(subset=[target_col_name], inplace=True)
            
            y = final_df[target_col_name]
            X = final_df.drop(columns=[target_col_name])
            
            # Ensure target column not in features
            if target_col_name in X.columns:
                X = X.drop(columns=[target_col_name])
            
            st.success(f"Generated {X.shape[1]} total features after feature engineering!")
        
        # Feature selection with RFE
        with st.spinner("Performing feature selection..."):
            # Split data
            test_size_ratio = st.session_state.test_size
            split_point = int(len(X) * (1 - test_size_ratio))
            
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Select features
            n_features = st.session_state.n_features
            selected_features, feature_ranking = select_features_with_rfe(
                X_train, y_train, n_features_to_select=n_features, target_name=target_col_name
            )
            
            if not selected_features:
                st.error("Feature selection failed. Please try different parameters.")
                return
            
            # Display top features
            st.subheader("Selected Features")
            st.dataframe(
                feature_ranking.head(min(n_features, len(feature_ranking))),
                use_container_width=True,
                hide_index=True
            )
            
            # Feature importance visualization
            fig = px.bar(
                feature_ranking.head(min(20, len(feature_ranking))), 
                x='Rank', y='Feature',
                title=f'Top {min(20, len(feature_ranking))} Features by Importance',
                orientation='h',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Train and evaluate model
        with st.spinner("Training and evaluating model..."):
            model, predictions, metrics, feature_importance = train_and_evaluate_model(
                X_train, y_train, X_test, y_test, selected_features, target_name=target_col_name
            )
            
            if model is None:
                st.error("Model training failed. Please try different parameters.")
                return
            
            # Save model and results to session state
            model_id = f"{selected_entity}_{selected_flow}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state['current_model'] = {
                'id': model_id,
                'entity': selected_entity,
                'flow': selected_flow,
                'entity_type': entity_type,
                'model': model,
                'predictions': predictions,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'selected_features': selected_features,
                'train_period': f"{X_train.index.min().strftime('%Y-%m-%d')} to {X_train.index.max().strftime('%Y-%m-%d')}",
                'test_period': f"{X_test.index.min().strftime('%Y-%m-%d')} to {X_test.index.max().strftime('%Y-%m-%d')}"
            }
            
            # Save model file
            model_path = models_dir / f"{model_id}.joblib"
            joblib.dump(model, model_path)
            
            # Save results
            results_path = results_dir / f"{model_id}_results.joblib"
            results_to_save = {
                'entity': selected_entity,
                'flow': selected_flow,
                'entity_type': entity_type,
                'predictions': predictions,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'selected_features': selected_features,
                'train_period': f"{X_train.index.min().strftime('%Y-%m-%d')} to {X_train.index.max().strftime('%Y-%m-%d')}",
                'test_period': f"{X_test.index.min().strftime('%Y-%m-%d')} to {X_test.index.max().strftime('%Y-%m-%d')}"
            }
            joblib.dump(results_to_save, results_path)
        
                # Generate Naive Forecast using the SAME y_train and y_test
        # After the existing model training code (around line 1142 in your file), add:

        # Generate Seasonal Naive Forecast using the SAME y_train and y_test
        st.subheader("Baseline Comparison")
        y_pred_naive = generate_naive_forecast(y_train, y_test.index)
        y_pred_seasonal = generate_seasonal_naive_forecast(y_train, y_test.index)

        # Calculate Naive Metrics
        naive_metrics = {}
        seasonal_metrics = {}
        try:
            # Ensure y_test and baseline predictions are aligned and non-empty
            common_idx = y_test.index.intersection(y_pred_naive.index)
            y_test_aligned = y_test[common_idx]
            y_pred_naive_aligned = y_pred_naive[common_idx]
            y_pred_seasonal_aligned = y_pred_seasonal[common_idx]

            if not y_test_aligned.empty:
                # Calculate metrics for naive forecast
                naive_metrics['RMSE'] = np.sqrt(mean_squared_error(y_test_aligned, y_pred_naive_aligned))
                naive_metrics['MAE'] = mean_absolute_error(y_test_aligned, y_pred_naive_aligned)
                naive_metrics['R²'] = r2_score(y_test_aligned, y_pred_naive_aligned)

                # Calculate metrics for seasonal naive forecast
                seasonal_metrics['RMSE'] = np.sqrt(mean_squared_error(y_test_aligned, y_pred_seasonal_aligned))
                seasonal_metrics['MAE'] = mean_absolute_error(y_test_aligned, y_pred_seasonal_aligned)
                seasonal_metrics['R²'] = r2_score(y_test_aligned, y_pred_seasonal_aligned)

                # Directional Accuracy for Naive
                actual_direction_naive = np.sign(y_test_aligned.diff().dropna())
                pred_direction_naive = np.sign(y_pred_naive_aligned.diff().dropna())
                common_indices_naive = actual_direction_naive.index.intersection(pred_direction_naive.index)

                # Directional Accuracy for Seasonal
                pred_direction_seasonal = np.sign(y_pred_seasonal_aligned.diff().dropna())
                common_indices_seasonal = actual_direction_naive.index.intersection(pred_direction_seasonal.index)

                if not common_indices_naive.empty:
                    match_naive = (actual_direction_naive[common_indices_naive] == pred_direction_naive[common_indices_naive])
                    naive_metrics['Directional Accuracy (%)'] = np.mean(match_naive) * 100 if not match_naive.empty else 0.0
                else:
                    naive_metrics['Directional Accuracy (%)'] = 0.0 # Or np.nan

                if not common_indices_seasonal.empty:
                    match_seasonal = (actual_direction_naive[common_indices_seasonal] == pred_direction_seasonal[common_indices_seasonal])
                    seasonal_metrics['Directional Accuracy (%)'] = np.mean(match_seasonal) * 100 if not match_seasonal.empty else 0.0
                else:
                    seasonal_metrics['Directional Accuracy (%)'] = 0.0 # Or np.nan

            else:
                st.warning("Could not align y_test and baseline predictions for metric calculation.")
                baseline_msg = {k: np.nan for k in ['RMSE', 'MAE', 'R²', 'Directional Accuracy (%)']}
                naive_metrics = seasonal_metrics = baseline_msg

        except Exception as e:
            st.error(f"Error calculating baseline metrics: {e}")
            baseline_msg = {k: np.nan for k in ['RMSE', 'MAE', 'R²', 'Directional Accuracy (%)']}
            naive_metrics = seasonal_metrics = baseline_msg

        # Display results including Baselines
        st.subheader("Model Performance")

        # Create a DataFrame to store all model metrics for comparison
        model_comparison = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²', 'Directional Accuracy (%)'],
            'XGBoost': [metrics.get('RMSE', np.nan), 
                        metrics.get('MAE', np.nan), 
                        metrics.get('R²', np.nan), 
                        metrics.get('Directional Accuracy (%)', np.nan)],
            'Naive': [naive_metrics.get('RMSE', np.nan), 
                    naive_metrics.get('MAE', np.nan), 
                    naive_metrics.get('R²', np.nan), 
                    naive_metrics.get('Directional Accuracy (%)', np.nan)],
            'Seasonal Naive': [seasonal_metrics.get('RMSE', np.nan), 
                            seasonal_metrics.get('MAE', np.nan), 
                            seasonal_metrics.get('R²', np.nan), 
                            seasonal_metrics.get('Directional Accuracy (%)', np.nan)]
        })

        # Format the numbers in the DataFrame for better display
        formatted_comparison = model_comparison.copy()
        for col in ['XGBoost', 'Naive', 'Seasonal Naive']:
            formatted_comparison[col] = formatted_comparison.apply(
                lambda x: f"{x[col]:.4f}" if x['Metric'] in ['RMSE', 'MAE', 'R²'] else f"{x[col]:.1f}%", 
                axis=1
            )

        # Display the comparison table
        st.dataframe(formatted_comparison.set_index('Metric'), use_container_width=True)

        # Show individual metrics with improvement percentages over naive baseline
        st.write("**XGBoost Model Performance:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            improvement_rmse = ((naive_metrics.get('RMSE', 0) - metrics.get('RMSE', 0)) / naive_metrics.get('RMSE', 1)) * 100
            st.metric("RMSE", f"{metrics.get('RMSE', np.nan):.4f}", 
                    f"{improvement_rmse:.1f}% vs Naive" if not np.isnan(improvement_rmse) else "")
        with col2:
            improvement_mae = ((naive_metrics.get('MAE', 0) - metrics.get('MAE', 0)) / naive_metrics.get('MAE', 1)) * 100
            st.metric("MAE", f"{metrics.get('MAE', np.nan):.4f}", 
                    f"{improvement_mae:.1f}% vs Naive" if not np.isnan(improvement_mae) else "")
        with col3:
            improvement_r2 = (metrics.get('R²', 0) - naive_metrics.get('R²', 0)) * 1
            st.metric("R²", f"{metrics.get('R²', np.nan):.4f}", 
                    f"{improvement_r2:.1f} pts vs Naive" if not np.isnan(improvement_r2) else "")
        with col4:
            improvement_dir = metrics.get('Directional Accuracy (%)', 0) - naive_metrics.get('Directional Accuracy (%)', 0)
            st.metric("Directional Accuracy", f"{metrics.get('Directional Accuracy (%)', np.nan):.1f}%", 
                    f"{improvement_dir:.1f} % vs Naive" if not np.isnan(improvement_dir) else "")
            
        # Create train/test visualization
        predictions['Date'] = predictions.index
        
        fig = px.line(
            predictions, 
            x='Date', 
            y=['Actual', 'Predicted'],
            color='Set',
            title=f"{target_col_name} - Actual vs Predicted",
            template="plotly_white"
        )
        
        # Add vertical line at train/test split
        split_date = X_train.index.max()
     
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance visualization
        fig = px.bar(
            feature_importance.head(min(20, len(feature_importance))),
            x='Importance',
            y='Feature',
            title=f'Top {min(20, len(feature_importance))} Features by XGBoost Importance',
            orientation='h',
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show residual analysis
        st.subheader("Residual Analysis")
        
        # Calculate residuals
        test_predictions = predictions[predictions['Set'] == 'Test'].copy()
        test_predictions['Residual'] = test_predictions['Actual'] - test_predictions['Predicted']
        test_predictions['Residual_Pct'] = (test_predictions['Residual'] / test_predictions['Actual']) * 100
        
        # Create residual plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                test_predictions,
                x='Predicted',
                y='Residual',
                title='Residual vs Predicted',
                template="plotly_white"
            )
            fig.add_hline(y=0, line_width=1, line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                test_predictions,
                x='Residual',
                title='Residual Distribution',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Residuals over time
        fig = px.line(
            test_predictions,
            x='Date',
            y='Residual',
            title='Residuals Over Time',
            template="plotly_white"
        )
        fig.add_hline(y=0, line_width=1, line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Success message
        st.success(f"Forecast model for {target_col_name} has been successfully created and saved!")

def main_backtesting():
    """Backtesting page functionality to evaluate forecast model performance over historical periods."""
    st.title("📊 Backtesting")
    
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        st.warning("Data not loaded. Please load data first.")
        return
        
    st.subheader("Evaluate Model Performance Over Historical Periods")
    
    # Step 1: Select entity and flow for backtesting
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        source_type = st.radio("Select Source Type:", 
                              ["Country (Supply Demand)", "Region (Regional Balance)"], 
                              index=0,
                              help="Choose between country-level or regional-level data", key='back')
    
    # Determine entity type and options based on selection
    if source_type == "Country (Supply Demand)":
        entity_type = "supply"
        entity_label = "Country"
        entity_options = st.session_state.countries
        flow_options = st.session_state.supply_flows
    else:
        entity_type = "regional"
        entity_label = "Region"
        entity_options = st.session_state.regions
        flow_options = st.session_state.regional_flows
    
    with col2:
        selected_entity = st.selectbox(f"Select {entity_label}:", entity_options, key="bt_entity")
    
    with col3:
        selected_flow = st.selectbox("Select Flow:", flow_options, key="bt_flow")
    
    # Step 2: Configure backtesting parameters
    st.subheader("Backtesting Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Number of historical windows to test
        n_windows = st.slider("Number of Test Windows", 3, 12, 6, 
                           help="Number of historical periods to test forecast accuracy")
    
    with col2:
        # Size of each test window (months)
        window_size = st.slider("Test Window Size (months)", 1, 12, 3, 
                             help="Length of each forecast window in months")
    
    with col3:
        # Size of training data (months or years)
        train_size_unit = st.radio("Training Size Unit", ["Months", "Years"], index=1)
        if train_size_unit == "Months":
            train_size = st.slider("Training Size", 12, 60, 24)
        else:
            train_size = st.slider("Training Size (years)", 1, 5, 2) * 12
    
    # Choose models to include in backtest
    model_options = st.multiselect(
        "Models to Include in Backtest",
        options=["XGBoost", "Naive", "Seasonal Naive"],
        default=["XGBoost", "Naive", "Seasonal Naive"],
        help="Select which models to evaluate during backtesting"
    )
    
    # Choose number of features for XGBoost
    n_features = st.slider("Number of Features for XGBoost", 10, 50, 20)
    
    # Option to use HPO
    use_hpo = st.checkbox("Use Hyperparameter Optimization for XGBoost", value=False,
                       help="Slower but potentially more accurate")
    
    # Run backtest button
    if st.button("Run Backtest", type="primary", use_container_width=True):
        # Check that data exists
        supply_demand = st.session_state.supply_demand
        regional_balance = st.session_state.regional_balance
        maintenance = st.session_state.maintenance
        netback = st.session_state.netback
        
        with st.spinner(f"Checking availability of {selected_entity} {selected_flow} data..."):
            if entity_type == "supply":
                has_data = not supply_demand[(supply_demand['CountryName'] == selected_entity) & 
                                          (supply_demand['FlowBreakdown'] == selected_flow)].empty
            else:
                has_data = not regional_balance[(regional_balance['GroupName'] == selected_entity) & 
                                             (regional_balance['FlowBreakdown'] == selected_flow)].empty
        
        if not has_data:
            st.error(f"No data available for {selected_entity} {selected_flow}. Please select a different combination.")
            return
        
        # Extract target data
        with st.spinner("Extracting target data..."):
            target_df, target_col_name = extract_target(
                supply_demand, regional_balance, selected_entity, selected_flow, entity_type
            )
            
            if target_df.empty:
                st.error(f"Could not extract target data for {selected_entity} {selected_flow}.")
                return
                
            # Make sure we have enough history
            if len(target_df) < train_size + (n_windows * window_size):
                st.warning(f"Not enough historical data for requested backtest configuration. \n"
                         f"Need at least {train_size + (n_windows * window_size)} months, "
                         f"but only have {len(target_df)} months.")
                st.info("Try reducing the number of windows, window size, or training period.")
                return
                
            # Sort target data by date and get date range for display
            target_df = target_df.sort_index()
            date_range = f"{target_df.index.min().strftime('%Y-%m')} to {target_df.index.max().strftime('%Y-%m')}"
            st.success(f"Found {len(target_df)} months of historical data ({date_range})")
            
            # Plot the target time series
            fig = px.line(
                target_df, 
                x=target_df.index, 
                y=target_col_name,
                title=f"{target_col_name} Historical Data",
                labels={"value": "Value", "ReferenceDate": "Date"},
                template="plotly_white"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # ---- Main Backtesting Loop ----
        # Initialize result storage
        results = {
            'window_start': [],
            'window_end': [],
            'train_start': [],
            'train_end': []
        }
        
        # Initialize metrics for each model
        for model_name in model_options:
            results[f'{model_name}_RMSE'] = []
            results[f'{model_name}_MAE'] = []
            results[f'{model_name}_R2'] = []
            results[f'{model_name}_DA'] = []  # Directional Accuracy
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate window starting points
        # We need to leave enough data at the end for the test windows
        max_end_idx = len(target_df) - window_size
        # The earliest we can start is after train_size
        min_start_idx = train_size
        # Adjust if not enough data
        usable_range = max_end_idx - min_start_idx
        # Calculate step size to evenly distribute windows
        step_size = max(1, usable_range // n_windows)
        
        # Create window starting points
        window_start_indices = list(range(min_start_idx, max_end_idx, step_size))[:n_windows]
        
        # Run each backtest window
        for i, start_idx in enumerate(window_start_indices):
            window_progress = (i / len(window_start_indices)) * 100
            progress_bar.progress(int(window_progress))
            
            # Define test window
            test_start_idx = start_idx
            test_end_idx = min(test_start_idx + window_size, len(target_df))
            
            # Define training window (going back from test start)
            train_start_idx = max(0, test_start_idx - train_size)
            train_end_idx = test_start_idx
            
            # Get date ranges for this window
            train_range = (
                target_df.index[train_start_idx].strftime('%Y-%m'),
                target_df.index[train_end_idx-1].strftime('%Y-%m')
            )
            test_range = (
                target_df.index[test_start_idx].strftime('%Y-%m'),
                target_df.index[test_end_idx-1].strftime('%Y-%m')
            )
            
            status_text.text(f"Processing window {i+1}/{len(window_start_indices)}: "
                           f"Training {train_range[0]} to {train_range[1]}, "
                           f"Testing {test_range[0]} to {test_range[1]}")
            
            # Store window info
            results['window_start'].append(target_df.index[test_start_idx])
            results['window_end'].append(target_df.index[test_end_idx-1])
            results['train_start'].append(target_df.index[train_start_idx])
            results['train_end'].append(target_df.index[train_end_idx-1])
            
            # Extract the exact training and testing data for this window
            train_data = target_df.iloc[train_start_idx:train_end_idx]
            test_data = target_df.iloc[test_start_idx:test_end_idx]
            
            y_train = train_data[target_col_name]
            y_test = test_data[target_col_name]
            
            # If XGBoost selected, build feature set and train model
            if "XGBoost" in model_options:
                # Generate features similar to main forecasting function
                with st.spinner(f"Generating features for window {i+1}..."):
                    # Similar feature generation code from main_forecasting
                    # Entity-specific features (excluding target flow)
                    entity_features = generate_entity_specific_features(
                        supply_demand, regional_balance, selected_entity, selected_flow, entity_type
                    )
                    
                    # Default to some major countries and regions for features
                    default_countries = ['China', 'United States', 'Russia', 'Saudi Arabia']
                    default_countries = [c for c in default_countries if c in st.session_state.countries and c != selected_entity]
                    
                    default_regions = ['OECD Asia Oceania', 'Other Asia', 'China', 'Africa']
                    default_regions = [r for r in default_regions if r in st.session_state.regions and r != selected_entity]
                    
                    default_flows = ['REFINOBS', 'REFCAP', 'CLOSTLV', 'TOTIMPSB']
                    default_flows = [f for f in default_flows if f in flow_options and f != selected_flow]
                    
                    # Feature generation - use defaults for backtesting
                    country_features = generate_country_features(
                        supply_demand, default_countries[:3], default_flows[:3]
                    )
                    
                    region_features = generate_regional_features(
                        regional_balance, default_regions[:2], default_flows[:3]
                    )
                    
                    # Select major countries for maintenance
                    default_maint_countries = ['United States', 'China']
                    maint_countries = [c for c in default_maint_countries if c in maintenance['Country'].unique()]
                    
                    maintenance_features = generate_maintenance_features(
                        maintenance, maint_countries
                    )
                    
                    # Market features
                    market_features = generate_market_features(netback)
                    
                    # Combine all feature sets
                    feature_dfs = [
                        entity_features, country_features, region_features,
                        maintenance_features, market_features
                    ]
                    # Filter out empty DataFrames
                    feature_dfs = [df for df in feature_dfs if df is not None and not df.empty]
                    
                    # Combine target with features
                    combined_df = combine_data(target_df.iloc[:test_end_idx], feature_dfs)
                    
                    # Add time features
                    combined_df = generate_time_features(combined_df)
                    
                    # Add trend features
                    combined_df = generate_trend_features(combined_df)
                    
                    # Add derived features - use standard lag and MA windows
                    final_df = generate_derived_features(
                        combined_df, target_col_name, 
                        lags=[1, 3, 6, 12], 
                        windows=[3, 6, 12]
                    )
                    
                    # Prepare X and y
                    final_df = final_df.sort_index()
                    final_df.dropna(subset=[target_col_name], inplace=True)
                    
                    # Create train/test split based on window indices
                    X = final_df.drop(columns=[target_col_name])
                    y = final_df[target_col_name]
                    
                    # Get the train/test split based on dates
                    train_dates = train_data.index
                    test_dates = test_data.index
                    
                    X_train = X.loc[X.index.isin(train_dates)]
                    y_train = y.loc[y.index.isin(train_dates)]
                    X_test = X.loc[X.index.isin(test_dates)]
                    y_test = y.loc[y.index.isin(test_dates)]
                    
                    # Select features
                    selected_features, _ = select_features_with_rfe(
                        X_train, y_train, n_features_to_select=n_features, target_name=target_col_name
                    )
                
                # Train and evaluate XGBoost model
                with st.spinner(f"Training XGBoost model for window {i+1}..."):
                    model, predictions, metrics, _ = train_and_evaluate_model(
                        X_train, y_train, X_test, y_test, selected_features, 
                        target_name=target_col_name, use_hpo=use_hpo
                    )
                    
                    # Store metrics
                    results['XGBoost_RMSE'].append(metrics.get('RMSE', np.nan))
                    results['XGBoost_MAE'].append(metrics.get('MAE', np.nan))
                    results['XGBoost_R2'].append(metrics.get('R²', np.nan))
                    results['XGBoost_DA'].append(metrics.get('Directional Accuracy (%)', np.nan))
            
            # For all windows, calculate naive models
            if "Naive" in model_options:
                # Generate naive forecast
                naive_pred = generate_naive_forecast(y_train, y_test.index)
                
                # Calculate naive metrics
                if not naive_pred.empty and not y_test.empty:
                    common_idx = y_test.index.intersection(naive_pred.index)
                    
                    if len(common_idx) > 0:
                        y_test_aligned = y_test[common_idx]
                        naive_pred_aligned = naive_pred[common_idx]
                        
                        naive_rmse = np.sqrt(mean_squared_error(y_test_aligned, naive_pred_aligned))
                        naive_mae = mean_absolute_error(y_test_aligned, naive_pred_aligned)
                        naive_r2 = r2_score(y_test_aligned, naive_pred_aligned)
                        
                        # Calculate directional accuracy
                        actual_dir = np.sign(y_test_aligned.diff().dropna())
                        naive_dir = np.sign(naive_pred_aligned.diff().dropna())
                        common_dir_idx = actual_dir.index.intersection(naive_dir.index)
                        
                        if len(common_dir_idx) > 0:
                            naive_da = np.mean(actual_dir[common_dir_idx] == naive_dir[common_dir_idx]) * 100
                        else:
                            naive_da = np.nan
                            
                        results['Naive_RMSE'].append(naive_rmse)
                        results['Naive_MAE'].append(naive_mae)
                        results['Naive_R2'].append(naive_r2)
                        results['Naive_DA'].append(naive_da)
                    else:
                        # If no alignment possible, store NaNs
                        results['Naive_RMSE'].append(np.nan)
                        results['Naive_MAE'].append(np.nan)
                        results['Naive_R2'].append(np.nan)
                        results['Naive_DA'].append(np.nan)
                else:
                    # If forecasts empty, store NaNs
                    results['Naive_RMSE'].append(np.nan)
                    results['Naive_MAE'].append(np.nan)
                    results['Naive_R2'].append(np.nan)
                    results['Naive_DA'].append(np.nan)
            
            # Calculate seasonal naive if selected
            if "Seasonal Naive" in model_options:
                # Generate seasonal naive forecast
                seasonal_pred = generate_seasonal_naive_forecast(y_train, y_test.index)
                
                # Calculate seasonal naive metrics
                if not seasonal_pred.empty and not y_test.empty:
                    common_idx = y_test.index.intersection(seasonal_pred.index)
                    
                    if len(common_idx) > 0:
                        y_test_aligned = y_test[common_idx]
                        seasonal_pred_aligned = seasonal_pred[common_idx]
                        
                        seasonal_rmse = np.sqrt(mean_squared_error(y_test_aligned, seasonal_pred_aligned))
                        seasonal_mae = mean_absolute_error(y_test_aligned, seasonal_pred_aligned)
                        seasonal_r2 = r2_score(y_test_aligned, seasonal_pred_aligned)
                        
                        # Calculate directional accuracy
                        actual_dir = np.sign(y_test_aligned.diff().dropna())
                        seasonal_dir = np.sign(seasonal_pred_aligned.diff().dropna())
                        common_dir_idx = actual_dir.index.intersection(seasonal_dir.index)
                        
                        if len(common_dir_idx) > 0:
                            seasonal_da = np.mean(actual_dir[common_dir_idx] == seasonal_dir[common_dir_idx]) * 100
                        else:
                            seasonal_da = np.nan
                            
                        results['Seasonal Naive_RMSE'].append(seasonal_rmse)
                        results['Seasonal Naive_MAE'].append(seasonal_mae)
                        results['Seasonal Naive_R2'].append(seasonal_r2)
                        results['Seasonal Naive_DA'].append(seasonal_da)
                    else:
                        # If no alignment possible, store NaNs
                        results['Seasonal Naive_RMSE'].append(np.nan)
                        results['Seasonal Naive_MAE'].append(np.nan)
                        results['Seasonal Naive_R2'].append(np.nan)
                        results['Seasonal Naive_DA'].append(np.nan)
                else:
                    # If forecasts empty, store NaNs
                    results['Seasonal Naive_RMSE'].append(np.nan)
                    results['Seasonal Naive_MAE'].append(np.nan)
                    results['Seasonal Naive_R2'].append(np.nan)
                    results['Seasonal Naive_DA'].append(np.nan)
        
        # Finalize progress
        progress_bar.progress(100)
        status_text.text("Backtesting completed! Analyzing results...")
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Display results summary
        st.subheader("Backtesting Results")
        
        # Format the dates for display
        results_df['Period'] = results_df.apply(
            lambda x: f"{x['window_start'].strftime('%Y-%m')} to {x['window_end'].strftime('%Y-%m')}", 
            axis=1
        )
        
        # Create a summary dataframe with averages
        summary_metrics = ['RMSE', 'MAE', 'R2', 'DA']
        summary_data = []
        
        for model in model_options:
            model_metrics = {}
            model_metrics['Model'] = model
            
            for metric in summary_metrics:
                col_name = f"{model}_{metric}"
                if col_name in results_df.columns:
                    model_metrics[metric] = results_df[col_name].mean()
                else:
                    model_metrics[metric] = np.nan
            
            summary_data.append(model_metrics)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display the summary of all models
        st.write("**Average Performance Across All Windows:**")
        
        # Format the summary for better display
        formatted_summary = summary_df.copy()
        for col in ['RMSE', 'MAE', 'R2']:
            formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
        
        # Format DA as percentage
        if 'DA' in formatted_summary.columns:
            formatted_summary['DA'] = formatted_summary['DA'].apply(
                lambda x: f"{x:.2f}%" if not np.isnan(x) else "N/A"
            )
        
        formatted_summary.columns = ['Model', 'RMSE', 'MAE', 'R²', 'Dir. Accuracy']
        
        # Change column order for better display
        formatted_summary = formatted_summary[['Model', 'RMSE', 'MAE', 'R²', 'Dir. Accuracy']]
        
        # Display the summary table
        st.dataframe(formatted_summary.set_index('Model'), use_container_width=True)
        
        # Create visualization of metrics across windows
        st.subheader("Performance Across Test Windows")
        
        # Plot RMSE for each model across windows
        rmse_data = results_df[['Period'] + [f"{model}_RMSE" for model in model_options]]
        rmse_data_melted = rmse_data.melt(
            id_vars=['Period'],
            value_vars=[f"{model}_RMSE" for model in model_options],
            var_name='Model',
            value_name='RMSE'
        )
        
        # Clean up model names for display
        rmse_data_melted['Model'] = rmse_data_melted['Model'].apply(lambda x: x.split('_')[0])
        
        fig_rmse = px.line(
            rmse_data_melted,
            x='Period',
            y='RMSE',
            color='Model',
            title='RMSE by Model Across Test Windows',
            markers=True,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Plot R² for each model across windows
        r2_data = results_df[['Period'] + [f"{model}_R2" for model in model_options]]
        r2_data_melted = r2_data.melt(
            id_vars=['Period'],
            value_vars=[f"{model}_R2" for model in model_options],
            var_name='Model',
            value_name='R²'
        )
        
        # Clean up model names for display
        r2_data_melted['Model'] = r2_data_melted['Model'].apply(lambda x: x.split('_')[0])
        
        fig_r2 = px.line(
            r2_data_melted,
            x='Period',
            y='R²',
            color='Model',
            title='R² by Model Across Test Windows',
            markers=True,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # Directional accuracy visualization
        da_data = results_df[['Period'] + [f"{model}_DA" for model in model_options]]
        da_data_melted = da_data.melt(
            id_vars=['Period'],
            value_vars=[f"{model}_DA" for model in model_options],
            var_name='Model',
            value_name='Directional Accuracy (%)'
        )
        
        # Clean up model names for display
        da_data_melted['Model'] = da_data_melted['Model'].apply(lambda x: x.split('_')[0])
        
        fig_da = px.line(
            da_data_melted,
            x='Period',
            y='Directional Accuracy (%)',
            color='Model',
            title='Directional Accuracy by Model Across Test Windows',
            markers=True,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_da, use_container_width=True)
        
        # Performance improvement visualization - XGBoost vs Naive
        if "XGBoost" in model_options and "Naive" in model_options:
            st.subheader("XGBoost Performance Improvement vs. Naive")
            
            improvement_data = []
            for i, row in results_df.iterrows():
                if pd.notna(row['XGBoost_RMSE']) and pd.notna(row['Naive_RMSE']) and row['Naive_RMSE'] != 0:
                    rmse_improvement = ((row['Naive_RMSE'] - row['XGBoost_RMSE']) / row['Naive_RMSE']) * 100
                else:
                    rmse_improvement = np.nan
                    
                if pd.notna(row['XGBoost_R2']) and pd.notna(row['Naive_R2']):
                    r2_improvement = row['XGBoost_R2'] - row['Naive_R2']
                else:
                    r2_improvement = np.nan
                    
                improvement_data.append({
                    'Period': row['Period'],
                    'RMSE Improvement (%)': rmse_improvement,
                    'R² Improvement': r2_improvement
                })
            
            improvement_df = pd.DataFrame(improvement_data)
            
            # Create a bar chart showing improvement percentage
            fig_improvement = px.bar(
                improvement_df,
                x='Period',
                y='RMSE Improvement (%)',
                title='XGBoost RMSE Improvement Over Naive Forecast (%)',
                template="plotly_white",
                color='RMSE Improvement (%)',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            st.plotly_chart(fig_improvement, use_container_width=True)
        
        # Detailed results in an expandable section
        with st.expander("Detailed Results by Window", expanded=False):
            # Show the raw results for each window
            for i, row in results_df.iterrows():
                st.write(f"**Window {i+1}: {row['Period']}**")
                st.write(f"Training period: {row['train_start'].strftime('%Y-%m')} to {row['train_end'].strftime('%Y-%m')}")
                
                # Create a table of metrics for this window
                window_metrics = {}
                for model in model_options:
                    window_metrics[model] = {
                        'RMSE': row.get(f"{model}_RMSE", np.nan),
                        'MAE': row.get(f"{model}_MAE", np.nan),
                        'R²': row.get(f"{model}_R2", np.nan),
                        'Dir. Accuracy': f"{row.get(f'{model}_DA', np.nan):.2f}%"
                    }
                
                # Convert to DataFrame for display
                window_df = pd.DataFrame(window_metrics).T
                st.dataframe(window_df, use_container_width=True)
                
                st.markdown("---")
        
        # Save results
        st.session_state['backtest_results'] = {
            'entity': selected_entity,
            'flow': selected_flow,
            'entity_type': entity_type,
            'results_df': results_df,
            'summary_df': summary_df,
            'config': {
                'n_windows': n_windows,
                'window_size': window_size,
                'train_size': train_size,
                'model_options': model_options,
                'n_features': n_features,
                'use_hpo': use_hpo
            }
        }
        
        # Success message
        st.success(f"Backtesting completed successfully for {selected_entity} {selected_flow}!")

def main_feature_analysis():
    """Feature analysis page functionality."""
    st.title("🔍 Feature Analysis")
    
    if 'current_model' not in st.session_state:
        st.warning("No model available for feature analysis. Please generate a forecast first.")
        return
    
    model_info = st.session_state['current_model']
    
    st.subheader(f"Feature Analysis: {model_info['entity']} {model_info['flow']}")
    
    # Get feature information
    feature_importance = model_info['feature_importance'].copy()
    selected_features = model_info['selected_features']
    
    # Show feature importance
    st.subheader("Feature Importance Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        # Create feature groups by type
        feature_groups = {
            'Lagged Features': [f for f in selected_features if '_lag_' in f],
            'Moving Average Features': [f for f in selected_features if '_ma_' in f],
            'Time Features': [f for f in selected_features if any(x in f for x in ['month_', 'quarter_', 'year', 'Month', 'Quarter'])],
            'Trend Features': [f for f in selected_features if any(x in f for x in ['trend_', 'months_since', 'years_since'])],
            'Market Features': [f for f in selected_features if any(x in f for x in ['Crack', 'Margin', 'Spread', 'Crude'])],
            'Maintenance Features': [f for f in selected_features if any(x in f for x in ['Maintenance', 'Capacity_Offline'])],
            'Regular Flow Features': [f for f in selected_features if all(x not in f for x in ['_lag_', '_ma_', 'month_', 'quarter_', 'trend_', 'Crack', 'Maintenance'])]
        }
        
        # Calculate importance by group
        group_importance = {}
        
        for group_name, features in feature_groups.items():
            if features:
                group_imp = feature_importance[feature_importance['Feature'].isin(features)]['Importance'].sum()
                group_importance[group_name] = group_imp
        
        # Create group importance dataframe
        group_df = pd.DataFrame({
            'Feature Group': list(group_importance.keys()),
            'Importance': list(group_importance.values())
        }).sort_values('Importance', ascending=False)
        
        # Plot feature group importance
        fig = px.pie(
            group_df,
            values='Importance',
            names='Feature Group',
            title='Feature Importance by Type',
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation analysis
    st.subheader("Feature Correlation Analysis")
    
    if hasattr(st.session_state, 'final_df'):
        # Get correlation matrix for selected features only
        corr_data = st.session_state.final_df[selected_features + [model_info['entity'] + '_' + model_info['flow']]]
        correlation = corr_data.corr()
        
        # Plot correlation heatmap
        fig = px.imshow(
            correlation,
            text_auto='.2f',
            title='Feature Correlation Matrix',
            template="plotly_white",
            color_continuous_scale=px.colors.diverging.RdBu_r
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation with target
        target_corr = correlation[model_info['entity'] + '_' + model_info['flow']].drop(model_info['entity'] + '_' + model_info['flow']).sort_values(ascending=False)
        
        top_corr_df = pd.DataFrame({
            'Feature': target_corr.index,
            'Correlation with Target': target_corr.values
        })
        
        # Plot top correlations
        fig = px.bar(
            top_corr_df.head(20),
            x='Correlation with Target',
            y='Feature',
            title='Top 20 Features by Correlation with Target',
            orientation='h',
            color='Correlation with Target',
            color_continuous_scale=px.colors.sequential.Blues,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Detailed feature correlation data not available. Generate a new forecast to see correlation analysis.")
    
    # Feature deep dive - scatter plots of top features vs target
    st.subheader("Feature Relationship Explorer")
    
    if len(selected_features) > 0:
        selected_feature = st.selectbox(
            "Select feature to explore:",
            options=selected_features,
            index=0
        )
        
        if hasattr(st.session_state, 'final_df') and selected_feature in st.session_state.final_df.columns:
            target_col = model_info['entity'] + '_' + model_info['flow']
            scatter_data = st.session_state.final_df[[selected_feature, target_col]].copy()
            scatter_data.dropna(inplace=True)
            
            fig = px.scatter(
                scatter_data,
                x=selected_feature,
                y=target_col,
                trendline="ols",
                title=f'Relationship between {selected_feature} and Target',
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display stats
            correlation = scatter_data[[selected_feature, target_col]].corr().iloc[0, 1]
            
            st.metric("Correlation with Target", f"{correlation:.4f}")
            
            # Show feature over time
            scatter_data['Date'] = scatter_data.index
            
            fig = px.line(
                scatter_data,
                x='Date',
                y=[selected_feature, target_col],
                title=f'Time Series: {selected_feature} vs Target',
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Detailed feature data not available for selected feature.")
    else:
        st.info("No features selected for the current model.")

def main_data_explorer():
    """Data explorer page functionality."""
    st.title("🔎 Data Explorer")
    
    # Ensure data is loaded
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        st.warning("Data not loaded. Please load data first.")
        return
    
    # Get data
    supply_demand = st.session_state.supply_demand
    regional_balance = st.session_state.regional_balance
    maintenance = st.session_state.maintenance
    netback = st.session_state.netback
    
    # Create tabs for different datasets
    tab1, tab2, tab3, tab4 = st.tabs(["Supply Demand", "Regional Balance", "Maintenance", "Netback"])
    
    with tab1:
        st.subheader("Supply Demand Data Explorer")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_country = st.selectbox(
                "Select Country:",
                options=sorted(supply_demand['CountryName'].unique()),
                key="sd_country"
            )
        
        with col2:
            selected_flow = st.selectbox(
                "Select Flow Breakdown:",
                options=sorted(supply_demand['FlowBreakdown'].unique()),
                key="sd_flow"
            )
        
        # Filter data
        filtered_data = supply_demand[
            (supply_demand['CountryName'] == selected_country) &
            (supply_demand['FlowBreakdown'] == selected_flow)
        ].copy()
        
        if filtered_data.empty:
            st.warning(f"No data available for {selected_country} {selected_flow}.")
        else:
            # Convert date and sort
            filtered_data['ReferenceDate'] = pd.to_datetime(filtered_data['ReferenceDate'])
            filtered_data = filtered_data.sort_values('ReferenceDate')
            
            # Show data preview
            st.dataframe(filtered_data, use_container_width=True)
            
            # Plot time series
            fig = px.line(
                filtered_data,
                x='ReferenceDate',
                y='ObservedValue',
                title=f"{selected_country} {selected_flow} Time Series",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show stats
            st.subheader("Statistical Summary")
            
            stats = filtered_data['ObservedValue'].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
            
            with col2:
                st.metric("Std Dev", f"{stats['std']:.2f}")
            
            with col3:
                st.metric("Min", f"{stats['min']:.2f}")
            
            with col4:
                st.metric("Max", f"{stats['max']:.2f}")
            
            # Date range info
            st.info(f"Date Range: {filtered_data['ReferenceDate'].min().strftime('%Y-%m-%d')} to {filtered_data['ReferenceDate'].max().strftime('%Y-%m-%d')}")
            
            # Year-over-year analysis if enough data
            if len(filtered_data) > 13:  # Need at least 13 months for YoY
                filtered_data.set_index('ReferenceDate', inplace=True)
                filtered_data['YoY_Change'] = filtered_data['ObservedValue'].pct_change(periods=12) * 100
                filtered_data.reset_index(inplace=True)
                
                # Remove rows with NaN YoY values
                yoy_data = filtered_data.dropna(subset=['YoY_Change'])
                
                if not yoy_data.empty:
                    fig = px.bar(
                        yoy_data,
                        x='ReferenceDate',
                        y='YoY_Change',
                        title=f"{selected_country} {selected_flow} Year-over-Year Change (%)",
                        template="plotly_white"
                    )
                    
                    fig.update_traces(marker_color=yoy_data['YoY_Change'].apply(
                        lambda x: 'green' if x > 0 else 'red'
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Regional Balance Data Explorer")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_region = st.selectbox(
                "Select Region:",
                options=sorted(regional_balance['GroupName'].unique()),
                key="rb_region"
            )
        
        with col2:
            selected_flow = st.selectbox(
                "Select Flow Breakdown:",
                options=sorted(regional_balance['FlowBreakdown'].unique()),
                key="rb_flow"
            )
        
        # Filter data
        filtered_data = regional_balance[
            (regional_balance['GroupName'] == selected_region) &
            (regional_balance['FlowBreakdown'] == selected_flow)
        ].copy()
        
        if filtered_data.empty:
            st.warning(f"No data available for {selected_region} {selected_flow}.")
        else:
            # Convert date and sort
            filtered_data['ReferenceDate'] = pd.to_datetime(filtered_data['ReferenceDate'])
            filtered_data = filtered_data.sort_values('ReferenceDate')
            
            # Show data preview
            st.dataframe(filtered_data, use_container_width=True)
            
            # Plot time series
            fig = px.line(
                filtered_data,
                x='ReferenceDate',
                y='ObservedValue',
                title=f"{selected_region} {selected_flow} Time Series",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show stats
            st.subheader("Statistical Summary")
            
            stats = filtered_data['ObservedValue'].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
            
            with col2:
                st.metric("Std Dev", f"{stats['std']:.2f}")
            
            with col3:
                st.metric("Min", f"{stats['min']:.2f}")
            
            with col4:
                st.metric("Max", f"{stats['max']:.2f}")
    
    with tab3:
        st.subheader("Maintenance Data Explorer")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            selected_country = st.selectbox(
                "Select Country:",
                options=sorted(maintenance['Country'].unique()),
                key="m_country"
            )
        
        with col2:
            selected_type = st.selectbox(
                "Select Outage Type:",
                options=['All'] + sorted(maintenance['OutageType'].unique()),
                key="m_type"
            )
        
        # Filter data
        if selected_type == 'All':
            filtered_data = maintenance[maintenance['Country'] == selected_country].copy()
        else:
            filtered_data = maintenance[
                (maintenance['Country'] == selected_country) &
                (maintenance['OutageType'] == selected_type)
            ].copy()
        
        if filtered_data.empty:
            st.warning(f"No maintenance data available for {selected_country} {selected_type}.")
        else:
            # Convert dates
            filtered_data['StartDate'] = pd.to_datetime(filtered_data['StartDate'])
            filtered_data['EndDate'] = pd.to_datetime(filtered_data['EndDate'])
            filtered_data = filtered_data.sort_values('StartDate')
            
            # Show data preview
            st.dataframe(filtered_data, use_container_width=True)
            
            # Summary stats
            st.subheader("Maintenance Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Events", f"{len(filtered_data)}")
            
            with col2:
                st.metric("Avg Capacity Offline", f"{filtered_data['CapacityOffline'].mean():.2f}")
            
            with col3:
                avg_duration = (filtered_data['EndDate'] - filtered_data['StartDate']).mean().days
                st.metric("Avg Duration (Days)", f"{avg_duration:.1f}")
            
            # Plot capacity offline over time
            if len(filtered_data) > 0:
                # Create a date range from min start to max end
                min_date = filtered_data['StartDate'].min()
                max_date = filtered_data['EndDate'].max()
                
                # Ensure min_date and max_date have actual values
                if pd.notnull(min_date) and pd.notnull(max_date):
                    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
                    
                    # Initialize a DataFrame with zeros for each day
                    capacity_offline = pd.DataFrame(0, index=date_range, columns=['CapacityOffline'])
                    
                    # For each maintenance event, add its capacity offline to the appropriate days
                    for _, event in filtered_data.iterrows():
                        start = max(event['StartDate'], min_date)
                        end = min(event['EndDate'], max_date)
                        
                        if pd.notnull(start) and pd.notnull(end) and pd.notnull(event['CapacityOffline']):
                            days = pd.date_range(start=start, end=end)
                            for day in days:
                                if day in capacity_offline.index:
                                    capacity_offline.loc[day, 'CapacityOffline'] += event['CapacityOffline']
                    
                    # Monthly aggregation for clearer visualization
                    monthly_capacity = capacity_offline.resample('MS').mean()
                    
                    fig = px.line(
                        monthly_capacity,
                        x=monthly_capacity.index,
                        y='CapacityOffline',
                        title=f"{selected_country} - Monthly Average Capacity Offline",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Netback Margin Data Explorer")
        
        # Convert dates and sort
        netback_data = netback.copy()
        netback_data['Dates'] = pd.to_datetime(netback_data['Dates'])
        netback_data = netback_data.sort_values('Dates')
        
        # Select columns to display
        col1, col2 = st.columns(2)
        
        with col1:
            selected_products = st.multiselect(
                "Select Products:",
                options=[col for col in netback.columns if col != 'Dates'],
                default=['Dubai Crude', '92 Ron Gasoline', 'Diesel 0.5 ppm', 'Dubai Netback Margin'],
                key="net_products"
            )
        
        with col2:
            date_range = st.date_input(
                "Select Date Range:",
                value=(
                    netback_data['Dates'].min().date(),
                    netback_data['Dates'].max().date()
                ),
                key="net_dates"
            )
        
        # Apply date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            netback_data = netback_data[
                (netback_data['Dates'].dt.date >= start_date) &
                (netback_data['Dates'].dt.date <= end_date)
            ]
        
        # Show data preview
        st.dataframe(netback_data[['Dates'] + selected_products], use_container_width=True)
        
        # Plot selected products
        if selected_products:
            # Create a long-format DataFrame for plotting
            plot_data = netback_data[['Dates'] + selected_products].melt(
                id_vars=['Dates'],
                value_vars=selected_products,
                var_name='Product',
                value_name='Price'
            )
            
            fig = px.line(
                plot_data,
                x='Dates',
                y='Price',
                color='Product',
                title="Selected Price Series",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and plot cracks/spreads
            if 'Dubai Crude' in selected_products and any(p in selected_products for p in ['92 Ron Gasoline', 'Diesel 0.5 ppm', 'Jet Fuel']):
                st.subheader("Crack Spreads")
                
                crack_data = pd.DataFrame({'Date': netback_data['Dates']})
                
                if '92 Ron Gasoline' in selected_products:
                    crack_data['Gasoline Crack'] = netback_data['92 Ron Gasoline'] - netback_data['Dubai Crude']
                
                if 'Diesel 0.5 ppm' in selected_products:
                    crack_data['Diesel Crack'] = netback_data['Diesel 0.5 ppm'] - netback_data['Dubai Crude']
                
                if 'Jet Fuel' in selected_products:
                    crack_data['Jet Crack'] = netback_data['Jet Fuel'] - netback_data['Dubai Crude']
                
                # Plot cracks
                if len(crack_data.columns) > 1:  # At least one crack + Date column
                    plot_data = crack_data.melt(
                        id_vars=['Date'],
                        value_vars=[col for col in crack_data.columns if col != 'Date'],
                        var_name='Crack',
                        value_name='Value'
                    )
                    
                    fig = px.line(
                        plot_data,
                        x='Date',
                        y='Value',
                        color='Crack',
                        title="Crack Spreads",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display statistics
            st.subheader("Statistical Summary")
            
            stats_df = netback_data[selected_products].describe().T
            stats_df = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            stats_df = stats_df.round(2)
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Show monthly averages for smoother view
            st.subheader("Monthly Averages")
            
            netback_data['Month'] = netback_data['Dates'].dt.to_period('M')
            monthly_data = netback_data.groupby('Month')[selected_products].mean().reset_index()
            monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
            
            plot_data = monthly_data.melt(
                id_vars=['Month'],
                value_vars=selected_products,
                var_name='Product',
                value_name='Price'
            )
            
            fig = px.line(
                plot_data,
                x='Month',
                y='Price',
                color='Product',
                title="Monthly Average Prices",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def main_model_management():
    """Model management page functionality."""
    st.title("📚 Model Management")
    
    # Get list of saved models
    model_files = list(models_dir.glob("*.joblib"))
    result_files = list(results_dir.glob("*_results.joblib"))
    
    if not model_files:
        st.warning("No saved models found. Generate forecasts to create models.")
        return
    
    # Create lookup from model ID to results file
    results_lookup = {}
    for result_file in result_files:
        model_id = result_file.stem.replace("_results", "")
        results_lookup[model_id] = result_file
    
    # Create table of models
    model_data = []
    
    for model_file in model_files:
        model_id = model_file.stem
        
        if model_id in results_lookup:
            # Load results without loading full model
            results = joblib.load(results_lookup[model_id])
            
            # Extract key info
            entity = results.get('entity', 'Unknown')
            flow = results.get('flow', 'Unknown')
            entity_type = results.get('entity_type', 'Unknown')
            train_period = results.get('train_period', 'Unknown')
            test_period = results.get('test_period', 'Unknown')
            
            # Extract metrics
            metrics = results.get('metrics', {})
            rmse = metrics.get('RMSE', float('nan'))
            r2 = metrics.get('R²', float('nan'))
            
            model_data.append({
                'ID': model_id,
                'Entity': entity,
                'Flow': flow,
                'Type': entity_type,
                'Train Period': train_period,
                'Test Period': test_period,
                'RMSE': rmse,
                'R²': r2,
                'Created': datetime.datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    # Display models table
    if model_data:
        model_df = pd.DataFrame(model_data)
        st.dataframe(model_df, use_container_width=True)
        
        # Select model to view
        selected_model_id = st.selectbox(
            "Select Model to View:",
            options=model_df['ID'].tolist(),
            index=0
        )
        
        if selected_model_id:
            st.divider()
            
            # Load model results
            results_file = results_lookup.get(selected_model_id)
            
            if results_file and results_file.exists():
                results = joblib.load(results_file)
                
                # Display model details
                st.subheader(f"Model: {results['entity']} {results['flow']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Entity Type:** {results['entity_type']}")
                    st.write(f"**Train Period:** {results['train_period']}")
                    st.write(f"**Test Period:** {results['test_period']}")
                
                with col2:
                    metrics = results.get('metrics', {})
                    st.metric("RMSE", f"{metrics.get('RMSE', 'N/A'):.4f}")
                    st.metric("R²", f"{metrics.get('R²', 'N/A'):.4f}")
                    st.metric("Directional Accuracy", f"{metrics.get('Directional Accuracy (%)', 'N/A'):.2f}%")
                
                # Display predictions
                st.subheader("Model Performance")
                
                predictions = results.get('predictions')
                
                if isinstance(predictions, pd.DataFrame):
                    # Create visualization
                    predictions['Date'] = predictions.index
                    
                    fig = px.line(
                        predictions,
                        x='Date',
                        y=['Actual', 'Predicted'],
                        color='Set',
                        title=f"{results['entity']} {results['flow']} - Actual vs Predicted",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show feature importance
                st.subheader("Feature Importance")
                
                feature_importance = results.get('feature_importance')
                
                if isinstance(feature_importance, pd.DataFrame):
                    st.dataframe(
                        feature_importance.head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Plot feature importance
                    fig = px.bar(
                        feature_importance.head(20),
                        x='Importance',
                        y='Feature',
                        title="Top 20 Features by Importance",
                        orientation='h',
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model actions
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Load Model to Current Session", type="primary", use_container_width=True):
                        # Load full model
                        model_file = models_dir / f"{selected_model_id}.joblib"
                        if model_file.exists():
                            try:
                                model = joblib.load(model_file)
                                
                                # Set as current model
                                st.session_state['current_model'] = {
                                    'id': selected_model_id,
                                    'entity': results['entity'],
                                    'flow': results['flow'],
                                    'entity_type': results['entity_type'],
                                    'model': model,
                                    'predictions': results['predictions'],
                                    'metrics': results['metrics'],
                                    'feature_importance': results['feature_importance'],
                                    'selected_features': results['selected_features'],
                                    'train_period': results['train_period'],
                                    'test_period': results['test_period']
                                }
                                
                                st.success(f"Model {selected_model_id} loaded successfully!")
                                
                            except Exception as e:
                                st.error(f"Error loading model: {e}")
                        else:
                            st.error(f"Model file not found: {model_file}")
                
                with col2:
                    if st.button("Delete Model", type="secondary", use_container_width=True):
                        try:
                            # Delete model and results files
                            model_file = models_dir / f"{selected_model_id}.joblib"
                            results_file = results_lookup.get(selected_model_id)
                            
                            if model_file.exists():
                                model_file.unlink()
                            
                            if results_file and results_file.exists():
                                results_file.unlink()
                            
                            st.success(f"Model {selected_model_id} deleted successfully!")
                            
                            # Refresh page
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error deleting model: {e}")
            else:
                st.error(f"Results file not found for model {selected_model_id}")
    else:
        st.warning("No model data available.")

def main():
    """Main application entry point."""
    st.sidebar.title("🛢️ Oil Flow Forecaster")
    
    # Load data if not already loaded
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            supply_demand, regional_balance, maintenance, netback = load_data()
            
            if supply_demand is not None and regional_balance is not None:
                # Store data in session state
                st.session_state.supply_demand = supply_demand
                st.session_state.regional_balance = regional_balance
                st.session_state.maintenance = maintenance
                st.session_state.netback = netback
                
                # Extract unique values
                st.session_state.countries, st.session_state.supply_flows = get_unique_values(
                    supply_demand, 'CountryName', 'FlowBreakdown'
                )
                
                st.session_state.regions, st.session_state.regional_flows = get_unique_values(
                    regional_balance, 'GroupName', 'FlowBreakdown'
                )
                
                # Set default feature engineering settings
                if 'lag_options' not in st.session_state:
                    st.session_state.lag_options = [1, 3, 6, 12]
                
                if 'ma_options' not in st.session_state:
                    st.session_state.ma_options = [3, 6, 12]
                
                if 'n_features' not in st.session_state:
                    st.session_state.n_features = 30
                
                if 'test_size' not in st.session_state:
                    st.session_state.test_size = 0.2
                
                st.session_state.data_loaded = True
    
    # Set up sidebar
    show_sidebar()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Forecasting", 
        "📊 Backtesting", 
        "🔍 Feature Analysis", 
        "🔎 Data Explorer",
        "📚 Model Management"
    ])
    
    # Check if data is loaded before rendering content
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        with tab1:
            main_forecasting(
                st.session_state.supply_demand,
                st.session_state.regional_balance,
                st.session_state.maintenance,
                st.session_state.netback
            )
        
        with tab2:
            main_backtesting()
        
        with tab3:
            main_feature_analysis()
        
        with tab4:
            main_data_explorer()
        
        with tab5:
            main_model_management()
    else:
        st.error("Please wait for data to load or check for errors in data loading.")

if __name__ == "__main__":
    main() 
    
    