"""
ML Training DAG for Board Game NLP Pipeline

This DAG orchestrates the machine learning pipeline including data preparation,
model training, validation, and deployment to the model registry.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable

from pipelines.ml.data_preprocessor import DataPreprocessor
from pipelines.ml.model_trainer import ModelTrainer
from pipelines.ml.model_validator import ModelValidator
from pipelines.ml.model_deployer import ModelDeployer

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'catchup': False,
}

# DAG definition
dag = DAG(
    'boardgame_ml_training',
    default_args=default_args,
    description='Train and deploy ML models for board game sentiment analysis',
    schedule_interval='0 4 * * 0',  # Weekly on Sunday at 4 AM
    max_active_runs=1,
    tags=['ml', 'training', 'sentiment-analysis'],
)

def prepare_training_data(**context):
    """Prepare data for model training."""
    preprocessor = DataPreprocessor()
    
    # Get configuration
    raw_data_path = Variable.get("raw_data_path", default_var="s3://boardgame-nlp-raw/bgg_comments/")
    train_split = float(Variable.get("train_split", default_var=0.8))
    val_split = float(Variable.get("val_split", default_var=0.1))
    
    # Prepare data
    data_splits = preprocessor.prepare_data(
        raw_data_path=raw_data_path,
        train_split=train_split,
        val_split=val_split,
        output_dir='/tmp/training_data'
    )
    
    context['task_instance'].xcom_push(
        key='data_splits', 
        value=data_splits
    )
    return data_splits

def train_sentiment_model(**context):
    """Train the sentiment analysis model."""
    trainer = ModelTrainer()
    data_splits = context['task_instance'].xcom_pull(key='data_splits')
    
    # Get training configuration
    model_config = {
        'model_name': Variable.get("model_name", default_var="albert-base-v2"),
        'max_length': int(Variable.get("max_length", default_var=512)),
        'batch_size': int(Variable.get("batch_size", default_var=16)),
        'learning_rate': float(Variable.get("learning_rate", default_var=2e-5)),
        'num_epochs': int(Variable.get("num_epochs", default_var=3)),
    }
    
    # Train model
    training_result = trainer.train_model(
        data_splits=data_splits,
        model_config=model_config,
        output_dir='/tmp/models'
    )
    
    context['task_instance'].xcom_push(
        key='training_result', 
        value=training_result
    )
    return training_result

def train_absa_model(**context):
    """Train the ABSA model."""
    trainer = ModelTrainer()
    data_splits = context['task_instance'].xcom_pull(key='data_splits')
    
    # Train ABSA model
    absa_result = trainer.train_absa_model(
        data_splits=data_splits,
        model_config=model_config,
        output_dir='/tmp/models'
    )
    
    context['task_instance'].xcom_push(
        key='absa_result', 
        value=absa_result
    )
    return absa_result

def validate_models(**context):
    """Validate trained models."""
    validator = ModelValidator()
    training_result = context['task_instance'].xcom_pull(key='training_result')
    absa_result = context['task_instance'].xcom_pull(key='absa_result')
    
    # Validate sentiment model
    sentiment_validation = validator.validate_model(
        model_path=training_result['model_path'],
        test_data_path=training_result['test_data_path'],
        model_type='sentiment'
    )
    
    # Validate ABSA model
    absa_validation = validator.validate_model(
        model_path=absa_result['model_path'],
        test_data_path=absa_result['test_data_path'],
        model_type='absa'
    )
    
    validation_results = {
        'sentiment': sentiment_validation,
        'absa': absa_validation
    }
    
    context['task_instance'].xcom_push(
        key='validation_results', 
        value=validation_results
    )
    return validation_results

def deploy_models(**context):
    """Deploy models to model registry and serving layer."""
    deployer = ModelDeployer()
    training_result = context['task_instance'].xcom_pull(key='training_result')
    absa_result = context['task_instance'].xcom_pull(key='absa_result')
    validation_results = context['task_instance'].xcom_pull(key='validation_results')
    
    # Check if models meet deployment criteria
    min_accuracy = float(Variable.get("min_model_accuracy", default_var=0.85))
    
    if (validation_results['sentiment']['accuracy'] >= min_accuracy and 
        validation_results['absa']['accuracy'] >= min_accuracy):
        
        # Deploy models
        deployment_result = deployer.deploy_models(
            sentiment_model_path=training_result['model_path'],
            absa_model_path=absa_result['model_path'],
            validation_results=validation_results
        )
        
        context['task_instance'].xcom_push(
            key='deployment_result', 
            value=deployment_result
        )
        return deployment_result
    else:
        raise ValueError("Models do not meet minimum accuracy requirements")

def run_model_tests(**context):
    """Run comprehensive model tests."""
    from pipelines.ml.model_tester import ModelTester
    
    tester = ModelTester()
    deployment_result = context['task_instance'].xcom_pull(key='deployment_result')
    
    # Run tests
    test_results = tester.run_comprehensive_tests(
        model_endpoints=deployment_result['endpoints']
    )
    
    context['task_instance'].xcom_push(
        key='test_results', 
        value=test_results
    )
    return test_results

# Task definitions
prepare_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag,
)

train_sentiment_task = PythonOperator(
    task_id='train_sentiment_model',
    python_callable=train_sentiment_model,
    dag=dag,
)

train_absa_task = PythonOperator(
    task_id='train_absa_model',
    python_callable=train_absa_model,
    dag=dag,
)

validate_models_task = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models,
    dag=dag,
)

deploy_models_task = PythonOperator(
    task_id='deploy_models',
    python_callable=deploy_models,
    dag=dag,
)

test_models_task = PythonOperator(
    task_id='run_model_tests',
    python_callable=run_model_tests,
    dag=dag,
)

# Model performance monitoring
model_monitoring = BashOperator(
    task_id='model_monitoring',
    bash_command='python pipelines/monitoring/model_monitor.py',
    dag=dag,
)

# Task dependencies
prepare_data_task >> [train_sentiment_task, train_absa_task]
[train_sentiment_task, train_absa_task] >> validate_models_task
validate_models_task >> deploy_models_task >> test_models_task
deploy_models_task >> model_monitoring
