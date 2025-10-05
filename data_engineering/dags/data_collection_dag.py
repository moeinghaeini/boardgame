"""
Data Collection DAG for Board Game NLP Pipeline

This DAG orchestrates the collection of board game data from BGG API,
including data validation, cleaning, and storage in the data lake.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable

from pipelines.extractors.bgg_extractor import BGGDataExtractor
from pipelines.transformers.data_cleaner import DataCleaner
from pipelines.loaders.data_lake_loader import DataLakeLoader
from pipelines.validators.data_validator import DataValidator

# Default arguments
default_args = {
    'owner': 'data-engineering-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# DAG definition
dag = DAG(
    'boardgame_data_collection',
    default_args=default_args,
    description='Collect and process board game data from BGG API',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    max_active_runs=1,
    tags=['data-collection', 'bgg', 'boardgames'],
)

def extract_bgg_data(**context):
    """Extract data from BGG API."""
    extractor = BGGDataExtractor()
    
    # Get configuration from Airflow Variables
    top_games_count = int(Variable.get("top_games_count", default_var=10))
    max_comments_per_game = int(Variable.get("max_comments_per_game", default_var=1000))
    
    # Extract data
    raw_data = extractor.extract_data(
        top_games_count=top_games_count,
        max_comments_per_game=max_comments_per_game
    )
    
    # Store in XCom for next task
    context['task_instance'].xcom_push(key='raw_data_path', value=raw_data)
    return raw_data

def validate_raw_data(**context):
    """Validate the extracted raw data."""
    validator = DataValidator()
    raw_data_path = context['task_instance'].xcom_pull(key='raw_data_path')
    
    validation_result = validator.validate_raw_data(raw_data_path)
    
    if not validation_result['is_valid']:
        raise ValueError(f"Data validation failed: {validation_result['errors']}")
    
    context['task_instance'].xcom_push(
        key='validation_result', 
        value=validation_result
    )
    return validation_result

def clean_and_transform_data(**context):
    """Clean and transform the raw data."""
    cleaner = DataCleaner()
    raw_data_path = context['task_instance'].xcom_pull(key='raw_data_path')
    
    # Clean and transform data
    cleaned_data_path = cleaner.clean_data(
        input_path=raw_data_path,
        output_path='/tmp/cleaned_data.csv'
    )
    
    context['task_instance'].xcom_push(
        key='cleaned_data_path', 
        value=cleaned_data_path
    )
    return cleaned_data_path

def load_to_data_lake(**context):
    """Load processed data to data lake."""
    loader = DataLakeLoader()
    cleaned_data_path = context['task_instance'].xcom_pull(key='cleaned_data_path')
    
    # Load to data lake
    data_lake_path = loader.load_to_raw_layer(
        local_path=cleaned_data_path,
        s3_bucket='boardgame-nlp-raw',
        s3_key=f"bgg_comments/{datetime.now().strftime('%Y/%m/%d')}/comments.csv"
    )
    
    context['task_instance'].xcom_push(
        key='data_lake_path', 
        value=data_lake_path
    )
    return data_lake_path

def generate_data_quality_report(**context):
    """Generate data quality report."""
    from pipelines.monitoring.data_quality_reporter import DataQualityReporter
    
    reporter = DataQualityReporter()
    data_lake_path = context['task_instance'].xcom_pull(key='data_lake_path')
    validation_result = context['task_instance'].xcom_pull(key='validation_result')
    
    # Generate quality report
    quality_report = reporter.generate_report(
        data_path=data_lake_path,
        validation_result=validation_result
    )
    
    # Send to monitoring system
    reporter.send_to_monitoring(quality_report)
    
    return quality_report

# Task definitions
extract_task = PythonOperator(
    task_id='extract_bgg_data',
    python_callable=extract_bgg_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_raw_data',
    python_callable=validate_raw_data,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_and_transform_data',
    python_callable=clean_and_transform_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_to_data_lake',
    python_callable=load_to_data_lake,
    dag=dag,
)

quality_report_task = PythonOperator(
    task_id='generate_data_quality_report',
    python_callable=generate_data_quality_report,
    dag=dag,
)

# Task dependencies
extract_task >> validate_task >> clean_task >> load_task >> quality_report_task

# Data quality monitoring task (runs in parallel)
data_quality_check = BashOperator(
    task_id='data_quality_check',
    bash_command='python pipelines/monitoring/data_quality_check.py',
    dag=dag,
)

# Set up parallel execution
[load_task, data_quality_check] >> quality_report_task
