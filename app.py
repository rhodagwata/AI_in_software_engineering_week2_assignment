"""
Cardiovascular Disease Risk Prediction
SDG 3: Good Health and Well-Being

This project uses machine learning to predict cardiovascular disease risk
based on patient health metrics, enabling early intervention and prevention.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

class CardiovascularDiseasePredictor:
    """
    A machine learning model to predict cardiovascular disease risk.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def load_data(self, filepath=None):
        """
        Load cardiovascular disease dataset.
        If no file provided, generates synthetic data for demonstration.
        """
        if filepath:
            self.data = pd.read_csv(filepath)
        else:
            # Generate synthetic data for demonstration
            print("Generating synthetic cardiovascular disease dataset...")
            np.random.seed(42)
            n_samples = 1000
            
            # Generate features
            age = np.random.randint(30, 80, n_samples)
            gender = np.random.randint(0, 2, n_samples)  # 0: Female, 1: Male
            height = np.random.normal(170, 10, n_samples)
            weight = np.random.normal(75, 15, n_samples)
            
            # Calculate BMI
            bmi = weight / ((height/100) ** 2)
            
            # Blood pressure (systolic and diastolic)
            ap_hi = np.random.normal(120, 20, n_samples)
            ap_lo = np.random.normal(80, 10, n_samples)
            
            # Cholesterol: 1 - normal, 2 - above normal, 3 - well above normal
            cholesterol = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2])
            
            # Glucose: 1 - normal, 2 - above normal, 3 - well above normal
            gluc = np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.25, 0.15])
            
            # Lifestyle factors
            smoke = np.random.randint(0, 2, n_samples)
            alco = np.random.randint(0, 2, n_samples)
            active = np.random.randint(0, 2, n_samples)
            
            # Generate target based on risk factors
            risk_score = (
                (age - 30) * 0.02 +
                gender * 0.1 +
                (bmi - 25) * 0.05 +
                (ap_hi - 120) * 0.01 +
                (ap_lo - 80) * 0.02 +
                (cholesterol - 1) * 0.3 +
                (gluc - 1) * 0.25 +
                smoke * 0.3 +
                alco * 0.15 +
                (1 - active) * 0.2 +
                np.random.normal(0, 0.2, n_samples)
            )
            
            cardio = (risk_score > np.percentile(risk_score, 50)).astype(int)
            
            self.data = pd.DataFrame({
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'bmi': bmi,
                'ap_hi': ap_hi,
                'ap_lo': ap_lo,
                'cholesterol': cholesterol,
                'gluc': gluc,
                'smoke': smoke,
                'alco': alco,
                'active': active,
                'cardio': cardio
            })
            
        print(f"Dataset loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis.
        """
        print("\n=== Data Overview ===")
        print(self.data.head())
        print("\n=== Data Statistics ===")
        print(self.data.describe())
        print("\n=== Missing Values ===")
        print(self.data.isnull().sum())
        print(f"\n=== Class Distribution ===")
        print(self.data['cardio'].value_counts())
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        self.data['cardio'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
        axes[0, 0].set_title('Cardiovascular Disease Distribution')
        axes[0, 0].set_xlabel('Disease Present (0=No, 1=Yes)')
        axes[0, 0].set_ylabel('Count')
        
        # Age distribution by disease status
        self.data.boxplot(column='age', by='cardio', ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Disease Status')
        axes[0, 1].set_xlabel('Disease Present')
        axes[0, 1].set_ylabel('Age')
        
        # BMI distribution by disease status
        self.data.boxplot(column='bmi', by='cardio', ax=axes[1, 0])
        axes[1, 0].set_title('BMI Distribution by Disease Status')
        axes[1, 0].set_xlabel('Disease Present')
        axes[1, 0].set_ylabel('BMI')
        
        # Correlation heatmap
        correlation = self.data.corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/eda_visualization.png', dpi=300, bbox_inches='tight')
        print("\n✓ EDA visualization saved!")
        
    def preprocess_data(self):
        """
        Preprocess the data for machine learning.
        """
        print("\n=== Preprocessing Data ===")
        
        # Separate features and target
        X = self.data.drop('cardio', axis=1)
        y = self.data['cardio']
        
        self.feature_names = X.columns.tolist()
        
        # Split data into training and testing sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Testing set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train the machine learning model.
        """
        print(f"\n=== Training {model_type} Model ===")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        print("✓ Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.
        """
        print("\n=== Model Evaluation ===")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        
        # Visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        
        # Feature Importance (for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            axes[2].barh(range(len(indices)), importances[indices])
            axes[2].set_yticks(range(len(indices)))
            axes[2].set_yticklabels([self.feature_names[i] for i in indices])
            axes[2].set_xlabel('Importance')
            axes[2].set_title('Top 10 Feature Importances')
            axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation visualization saved!")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def predict_risk(self, patient_data):
        """
        Predict cardiovascular disease risk for new patient data.
        
        Args:
            patient_data: Dictionary with patient features
        """
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_df)
        
        # Make prediction
        prediction = self.model.predict(patient_scaled)[0]
        probability = self.model.predict_proba(patient_scaled)[0]
        
        result = {
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability_no_disease': probability[0],
            'probability_disease': probability[1]
        }
        
        return result


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("CARDIOVASCULAR DISEASE RISK PREDICTION")
    print("SDG 3: Good Health and Well-Being")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CardiovascularDiseasePredictor()
    
    # Load data
    data = predictor.load_data()
    
    # Explore data
    predictor.explore_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = predictor.preprocess_data()
    
    # Train model
    predictor.train_model(X_train, y_train, model_type='random_forest')
    
    # Evaluate model
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Example prediction for a new patient
    print("\n=== Example Patient Risk Prediction ===")
    example_patient = {
        'age': 55,
        'gender': 1,
        'height': 175,
        'weight': 85,
        'bmi': 27.8,
        'ap_hi': 140,
        'ap_lo': 90,
        'cholesterol': 2,
        'gluc': 1,
        'smoke': 1,
        'alco': 0,
        'active': 0
    }
    
    risk_result = predictor.predict_risk(example_patient)
    print(f"\nPatient Risk Level: {risk_result['risk_level']}")
    print(f"Probability of Disease: {risk_result['probability_disease']:.2%}")
    print(f"Probability of No Disease: {risk_result['probability_no_disease']:.2%}")
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETE - Check /mnt/user-data/outputs/ for visualizations")
    print("=" * 60)


if __name__ == "__main__":
    main()
