#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess(raw_path="namadataset_raw/wa_customer_churn_total.csv",
               output_dir="preprocessing/namadataset_preprocessing"):
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading raw data...")
    df = pd.read_csv(raw_path)
    print(f"[INFO] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop('customerID', axis=1, inplace=True)

    le_churn = LabelEncoder()
    df['Churn'] = le_churn.fit_transform(df['Churn'])

    ordinal_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in ordinal_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    nominal_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    bins = [0, 12, 24, 48, 72, float('inf')]
    labels = ['0-12', '13-24', '25-48', '49-72', '72+']
    df['TenureBin'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    df['TenureBin'] = LabelEncoder().fit_transform(df['TenureBin'].astype(str))
    df['SeniorPartner'] = df['SeniorCitizen'] * df['Partner']

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

    # Feature lists for scaling
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                        'ChargesPerMonth', 'TenureBin', 'SeniorPartner']

    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE: {len(X_resampled)} training samples")

    scaler = StandardScaler()
    X_train_scaled = X_resampled.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train_scaled[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test_scaled[numeric_features])

    X_train_scaled.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_resampled.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print(f"[INFO] Preprocessing complete. Files saved to {output_dir}/")
    return X_train_scaled, X_test_scaled, y_resampled, y_test, scaler

if __name__ == "__main__":
    preprocess()
