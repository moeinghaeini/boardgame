"""
Model training module for sentiment analysis.
"""

import torch
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AlbertForSequenceClassification, 
    AlbertTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction
)
from datasets import load_dataset, Dataset
import evaluate

from .utils import ConfigManager


class ModelTrainer:
    """Trains ALBERT model for sentiment analysis."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = config_manager.logger
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_dataset(self, dataset_name: str = 'imdb') -> Tuple[Dataset, Dataset]:
        """
        Load and prepare dataset for training.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        try:
            self.logger.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
            
            train_data = dataset['train']
            test_data = dataset['test']
            
            self.logger.info(f"Loaded {len(train_data)} training samples, {len(test_data)} test samples")
            
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer: AlbertTokenizer) -> Dataset:
        """
        Tokenize dataset for training.
        
        Args:
            dataset: Dataset to tokenize
            tokenizer: Tokenizer to use
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            return tokenizer(
                examples['text'], 
                padding="max_length", 
                truncation=True, 
                max_length=self.config.get('model.max_length', 512)
            )
        
        return dataset.map(tokenize_function, batched=True)
    
    def setup_model(self, num_labels: int = 2) -> Tuple[AlbertForSequenceClassification, AlbertTokenizer]:
        """
        Setup model and tokenizer for training.
        
        Args:
            num_labels: Number of classification labels
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = self.config.get('model.name', 'albert-base-v2')
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
            
            # Load model
            self.model = AlbertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels
            )
            
            self.logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of metrics
        """
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=1)
        
        # Load metrics
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        
        # Compute metrics
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
        precision = precision_metric.compute(
            predictions=predictions, 
            references=labels, 
            average='weighted'
        )['precision']
        recall = recall_metric.compute(
            predictions=predictions, 
            references=labels, 
            average='weighted'
        )['recall']
        f1 = f1_metric.compute(
            predictions=predictions, 
            references=labels, 
            average='weighted'
        )['f1']
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, 
              train_dataset: Dataset, 
              eval_dataset: Dataset,
              output_dir: Optional[str] = None) -> Trainer:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory for model
            
        Returns:
            Trained trainer object
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not setup. Call setup_model() first.")
        
        if output_dir is None:
            output_dir = self.config.get_model_path()
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy='epoch',
            learning_rate=self.config.get('model.learning_rate', 2e-5),
            per_device_train_batch_size=self.config.get('model.batch_size', 16),
            per_device_eval_batch_size=self.config.get('model.batch_size', 16) * 2,
            num_train_epochs=self.config.get('model.num_epochs', 3),
            weight_decay=self.config.get('model.weight_decay', 0.01),
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            logging_steps=100,
            logging_dir=f"{output_dir}/logs"
        )
        
        # Setup data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        self.logger.info("Starting model training...")
        self.trainer.train()
        
        self.logger.info("Training completed successfully")
        return self.trainer
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            eval_dataset: Evaluation dataset (uses trainer's eval dataset if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if eval_dataset is not None:
            self.trainer.eval_dataset = eval_dataset
        
        self.logger.info("Evaluating model...")
        results = self.trainer.evaluate()
        
        self.logger.info("Evaluation results:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_model(self, save_path: Optional[str] = None) -> None:
        """
        Save the trained model and tokenizer.
        
        Args:
            save_path: Path to save model (uses config path if None)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not available for saving")
        
        if save_path is None:
            save_path = self.config.get_model_path()
        
        try:
            self.logger.info(f"Saving model to {save_path}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def train_full_pipeline(self, dataset_name: str = 'imdb') -> Dict[str, float]:
        """
        Complete training pipeline.
        
        Args:
            dataset_name: Name of dataset to use for training
            
        Returns:
            Dictionary of final evaluation metrics
        """
        # Load dataset
        train_data, test_data = self.load_dataset(dataset_name)
        
        # Setup model
        self.setup_model()
        
        # Tokenize datasets
        train_tokenized = self.tokenize_dataset(train_data, self.tokenizer)
        test_tokenized = self.tokenize_dataset(test_data, self.tokenizer)
        
        # Train model
        self.train(train_tokenized, test_tokenized)
        
        # Evaluate model
        results = self.evaluate()
        
        # Save model
        self.save_model()
        
        return results
