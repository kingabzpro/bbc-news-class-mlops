#!/usr/bin/env python
"""
Feast Feature Repository Example

This file demonstrates how to set up a Feast feature store by:
1. Setting up feature store configuration
2. Defining feature views for machine learning
3. Creating different types of data sources
4. Setting up entities for feature joins
5. Creating feature services for serving features

To use this feature repository:
1. Modify the data sources and feature definitions as needed
2. Run `feast apply` to apply the definitions to your feature store
3. Use `feast materialize` to load feature values into your online store
4. Query features with `feast get-online-features` or the Python API

For more information: https://docs.feast.dev/
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

# Import Feast components
from feast import (
    Entity, FeatureService, FeatureView, Field, FileSource,
    PushSource, RequestSource, SnowflakeSource, SqliteSource,
    ValueType, EmbeddingFeatureView, OnDemandFeatureView
)
from feast.types import (Float32, Float64, Int32, Int64, String, 
                        Array, Bytes, Bool, UnixTimestamp)
from feast.infra.offline_stores.file_source import FileLoaderFormat, FileSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.data_source import PushSource
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.repo_config import RepoConfig
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ================================================================
# Feature Store Configuration
# ================================================================
"""
This section defines the feature store configuration.

The feature_store.yaml file is generated automatically when you run
`feast init` or it can be created manually. This Python equivalent
shows how you can configure the feature store programmatically.
"""

# Data directory for this example
data_dir = os.path.join(project_root, "data")

def get_feature_store_config():
    """
    Define the feature store configuration.
    
    This function returns a RepoConfig object that configures how Feast
    stores and retrieves features for both offline and online access.
    
    Returns:
        RepoConfig: Feature store configuration
    """
    # Define the project name
    project_name = "mlops-features"
    
    # Path to the registry file that stores feature definitions
    registry = os.path.join(project_root, "data", "feast", "registry.db")
    
    # Configure the online store (for low-latency feature serving)
    online_store = SqliteOnlineStoreConfig(
        path=os.path.join(data_dir, "feast", "online_store.db")
    )
    
    # Create and return the configuration
    return RepoConfig(
        project=project_name,
        registry=registry,
        provider="local",  # Use local file system for offline store
        online_store=online_store,
        entity_key_serialization_version=2,  # Use v2 serialization
        # Optionally, add more configurations:
        # offline_store=...,  # For production, use Redshift, BigQuery, Snowflake, etc.
        # flags={"enable_auth": False},
    )

# ================================================================
# Data Sources
# ================================================================
"""
This section defines the data sources that Feast will use to load feature values.
 
Feast supports multiple data source types, including:
1. FileSource - Load data from files (CSV, Parquet)
2. SqliteSource - Load data from SQLite databases
3. SnowflakeSource - Load data from Snowflake
4. BigQuerySource - Load data from BigQuery
5. RedshiftSource - Load data from Redshift
6. PushSource - For streaming features
7. RequestSource - For features computed on-demand at request time
"""

# Example parquet file with customer demographic data
customer_source = FileSource(
    name="customer_demographics",
    path=os.path.join(data_dir, "customers.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Customer demographic features",
    # Uncomment when using a file in S3/GCS:
    # file_format=FileLoaderFormat.PARQUET,
    # s3_endpoint_override="https://s3.amazonaws.com",
)

# Example CSV file with transaction history data
transaction_source = FileSource(
    name="customer_transactions",
    path=os.path.join(data_dir, "transactions.csv"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="Customer transaction history features",
    file_format=FileLoaderFormat.CSV, 
)

# Example SQLite source for product catalog data
product_source = SqliteSource(
    name="product_catalog",
    query="SELECT * FROM products",
    timestamp_field="event_timestamp",
    database=os.path.join(data_dir, "products.db"),
    description="Product features from catalog database",
)

# Example Snowflake source (commented out - for reference)
"""
snowflake_source = SnowflakeSource(
    name="inventory_features",
    database="ANALYTICS",
    schema="ML_FEATURES",
    table="INVENTORY_FEATURES",
    timestamp_field="EVENT_TIMESTAMP",
    created_timestamp_column="CREATED_TIMESTAMP",
    query="SELECT * FROM ANALYTICS.ML_FEATURES.INVENTORY_FEATURES",
    description="Inventory features from Snowflake",
)
"""

# Example push source for real-time features
realtime_source = PushSource(
    name="realtime_features",
    batch_source=FileSource(
        path=os.path.join(data_dir, "realtime_features.parquet"),
        timestamp_field="event_timestamp",
    ),
    description="Real-time features pushed from applications",
)

# ================================================================
# Entities
# ================================================================
"""
Entities are objects in your domain that features are associated with.

For example, in a retail setting, entities might include:
- customers
- products
- stores
- transactions

Entities are used to join features from different feature views.
"""

# Customer entity
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Customer identifier",
    # Optional: tags for categorization and documentation
    tags={"category": "identity", "owner": "customer_data_team"},
)

# Product entity
product = Entity(
    name="product",
    join_keys=["product_id"],
    description="Product identifier",
    tags={"category": "item", "owner": "product_data_team"},
)

# Store entity
store = Entity(
    name="store",
    join_keys=["store_id"],
    description="Physical store location",
    tags={"category": "location", "owner": "retail_ops_team"},
)

# ================================================================
# Feature Views
# ================================================================
"""
Feature views are groups of features that share the same data source and entity.

Each feature view defines:
1. A data source to pull features from
2. An entity the features are associated with
3. A list of features (fields) to include
4. Time-to-live (TTL) for the features
"""

# Customer demographic features
customer_demographics_view = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        Field(name="age", dtype=Int32),
        Field(name="gender", dtype=String),
        Field(name="income_bracket", dtype=String),
        Field(name="customer_lifetime_value", dtype=Float32),
        Field(name="signup_date", dtype=String),
    ],
    online=True,  # Make available in online feature store
    source=customer_source,
    tags={"team": "customer_insights"},
    description="Demographic features for customers",
)

# Transaction features
customer_transactions_view = FeatureView(
    name="customer_transactions",
    entities=[customer],
    ttl=timedelta(days=90),  # Features valid for 90 days
    schema=[
        Field(name="average_transaction_value", dtype=Float32),
        Field(name="transaction_count_30d", dtype=Int32),
        Field(name="transaction_count_90d", dtype=Int32),
        Field(name="last_transaction_date", dtype=UnixTimestamp),
        Field(name="has_returned_item", dtype=Bool),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "transaction_analytics"},
    description="Features derived from customer transaction history",
)

# Product features
product_features_view = FeatureView(
    name="product_features",
    entities=[product],
    ttl=timedelta(days=180),  # Features valid for 180 days
    schema=[
        Field(name="price", dtype=Float32),
        Field(name="brand", dtype=String),
        Field(name="category", dtype=String),
        Field(name="rating", dtype=Float32),
        Field(name="stock_level", dtype=Int32),
    ],
    online=True,
    source=product_source,
    tags={"team": "product_analytics"},
    description="Features related to products in the catalog",
)

# ================================================================
# On-Demand Feature Views
# ================================================================
"""
On-demand feature views compute features on-the-fly during feature retrieval.

They're useful for:
1. Computing features that depend on request context
2. Creating transformations that combine features from multiple feature views
3. Implementing business logic that transforms features at serving time
"""

# Define an input schema for the on-demand feature view
@on_demand_feature_view(
    sources=[
        customer_demographics_view,
        customer_transactions_view,
    ],
    schema=[
        Field(name="recency_score", dtype=Float32),
        Field(name="frequency_score", dtype=Float32),
        Field(name="monetary_score", dtype=Float32),
        Field(name="rfm_score", dtype=Float32),
    ],
    description="Computed RFM (Recency, Frequency, Monetary) score for customers",
)
def customer_rfm_score(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) score for customers.
    
    Args:
        inputs: DataFrame containing input features
        
    Returns:
        DataFrame with computed RFM scores
    """
    df = pd.DataFrame()
    
    # Calculate recency score (based on last transaction date)
    # Higher score for more recent transactions
    current_time = datetime.now().timestamp()
    df["recency_score"] = 1.0 - ((current_time - inputs["last_transaction_date"]) 
                               / (90 * 24 * 60 * 60)) # Normalize by 90 days
    df["recency_score"] = df["recency_score"].clip(0, 1) * 5  # Scale to 0-5
    
    # Calculate frequency score (based on transaction count)
    df["frequency_score"] = inputs["transaction_count_90d"] / 10.0  # Normalize
    df["frequency_score"] = df["frequency_score"].clip(0, 5)  # Scale to 0-5
    
    # Calculate monetary score (based on average transaction value)
    df["monetary_score"] = inputs["average_transaction_value"] / 200.0  # Normalize
    df["monetary_score"] = df["monetary_score"].clip(0, 5)  # Scale to 0-5
    
    # Calculate combined RFM score
    df["rfm_score"] = (df["recency_score"] + df["frequency_score"] + df["monetary_score"]) / 3.0
    
    return df

# ================================================================
# Feature Services
# ================================================================
"""
Feature services are logical groupings of feature views that should
be accessed together for model training or inference.

They provide a convenient way to request a set of features for a specific use case.
"""

# Customer prediction feature service
customer_prediction_service = FeatureService(
    name="customer_prediction_features",
    features=[
        customer_demographics_view[["age", "income_bracket", "customer_lifetime_value"]],
        customer_transactions_view,
        customer_rfm_score,
    ],
    description="Features used for customer behavior prediction models",
    tags={"model": "customer_churn", "version": "1.0"},
)

# Product recommendation feature service
product_recommendation_service = FeatureService(
    name="product_recommendation_features",
    features=[
        customer_demographics_view, 
        customer_transactions_view[["average_transaction_value", "transaction_count_90d"]],
        product_features_view,
    ],
    description="Features used for product recommendation models",
    tags={"model": "product_recommendation", "version": "2.1"},
)

# ================================================================
# Example Usage
# ================================================================
"""
Below are examples showing how to use the feature store.

These functions demonstrate common operations but are not part of the
feature registry - they just provide usage examples.
"""

def example_get_historical_features():
    """
    Example showing how to get historical features for model training.
    """
    from feast import FeatureStore
    
    # Initialize the feature store
    store = FeatureStore(repo_path=".")
    
    # Define an entity DataFrame
    entity_df = pd.DataFrame({
        "customer_id": [1001, 1002, 1003, 1004, 1005],
        "event_timestamp": [
            datetime.now() - timedelta(days=1),
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=5),
        ]
    })
    
    # Retrieve historical features
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_demographics:age",
            "customer_demographics:income_bracket",
            "customer_transactions:average_transaction_value",
            "customer_transactions:transaction_count_90d",
            "customer_rfm_score:rfm_score",
        ],
    ).to_df()
    
    print(f"Retrieved training data with {len(training_df)} rows")
    return training_df

def example_get_online_features():
    """
    Example showing how to get online features for model inference.
    """
    from feast import FeatureStore
    
    # Initialize the feature store
    store = FeatureStore(repo_path=".")
    
    # Request online features
    features = store.get_online_features(
        features=[
            "customer_demographics:age",
            "customer_demographics:income_bracket",
            "customer_transactions:average_transaction_value",
            "customer_rfm_score:rfm_score", 
        ],
        entity_rows=[
            {"customer_id": 1001},
            {"customer_id": 1002},
        ],
    ).to_dict()
    
    print("Retrieved online features")
    return features

def example_use_feature_service():
    """
    Example showing how to use a feature service for batch retrieval.
    
    Feature services make it easy to consistently retrieve the same
    set of features for both training and inference.
    """
    from feast import FeatureStore
    
    # Initialize the feature store
    store = FeatureStore(repo_path=".")
    
    # Define an entity DataFrame
    entity_df = pd.DataFrame({
        "customer_id": [1001, 1002, 1003],
        "event_timestamp": [
            datetime.now() - timedelta(days=1),
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(days=3),
        ]
    })
    
    # Retrieve features using the customer prediction service
    # This will fetch all features defined in the service
    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=customer_prediction_service
    ).to_df()
    
    print(f"Retrieved {len(features_df.columns)} features using feature service")
    print(f"Features: {', '.join(features_df.columns)}")
    
    # You can also use feature services for online feature retrieval
    online_features = store.get_online_features(
        features=product_recommendation_service,
        entity_rows=[
            # Join with both customer and product entities
            {"customer_id": 1001, "product_id": 5001},
            {"customer_id": 1002, "product_id": 5002},
        ]
    ).to_dict()
    
    print("Retrieved features using feature service for online serving")
    return features_df

def materialize_features():
    """
    Example showing how to materialize features into the online store.
    
    This operation loads feature values from offline storage (files, databases)
    into the online feature store for low-latency serving.
    """
    from feast import FeatureStore
    from datetime import datetime, timedelta
    
    # Initialize the feature store
    store = FeatureStore(repo_path=".")
    
    # Define the time range to materialize
    start_time = datetime.now() - timedelta(days=30)
    end_time = datetime.now()
    
    # Materialize features - this will populate the online store
    store.materialize(
        start_date=start_time,
        end_date=end_time,
    )
    
    print(f"Materialized features from {start_time} to {end_time}")

# Example of how to run the Feast CLI commands
"""
# Apply feature definitions to the feature store
$ feast apply

# Register features in the registry
$ feast registry-dump

# Materialize feature values to the online store
$ feast materialize 2025-04-01T00:00:00 2025-05-07T00:00:00

# Get online features
$ feast get-online-features \
    --features customer_demographics:age,customer_transactions:average_transaction_value \
    --entity customer_id:1001

# List available feature views
$ feast feature-views list

# Teardown the feature store
$ feast teardown
"""

