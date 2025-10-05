#!/usr/bin/env python3
"""
Main script for Board Game NLP Analysis.

This script provides a command-line interface for running the complete
board game sentiment analysis pipeline.
"""

import argparse
import sys
from pathlib import Path

from boardgame_nlp import DataCollector, SentimentAnalyzer, ABSAAnalyzer, ModelTrainer
from boardgame_nlp.utils import ConfigManager
from boardgame_nlp.quality_metrics import QualityMetrics


def setup_directories(config: ConfigManager) -> None:
    """Create necessary directories."""
    from boardgame_nlp.utils import ensure_directory
    
    ensure_directory(config.get('data.base_path', './data'))
    ensure_directory(config.get('data.results_path', './Results'))
    ensure_directory(config.get('model.model_path', './Model'))


def collect_data(config: ConfigManager) -> None:
    """Collect board game data."""
    print("🔍 Collecting board game data...")
    
    collector = DataCollector(config)
    df = collector.collect_data()
    
    print(f"✅ Collected {len(df)} English comments")
    return df


def train_model(config: ConfigManager) -> None:
    """Train the sentiment analysis model."""
    print("🤖 Training sentiment analysis model...")
    
    trainer = ModelTrainer(config)
    results = trainer.train_full_pipeline('imdb')
    
    print("✅ Model training completed")
    print(f"📊 Final metrics: {results}")
    return results


def analyze_sentiment(config: ConfigManager, df) -> None:
    """Perform sentiment analysis on board game comments."""
    print("😊 Analyzing sentiment...")
    
    analyzer = SentimentAnalyzer(config)
    analyzer.load_model()
    
    # Analyze sentiment
    df_with_sentiment = analyzer.analyze_dataframe(df)
    
    # Save results
    output_path = config.get_data_path('sentiment_analysis_results.csv')
    analyzer.save_results(df_with_sentiment, output_path)
    
    # Generate quality report
    quality_metrics = QualityMetrics(config)
    quality_report_path = config.get_data_path('sentiment_quality_report.txt')
    quality_metrics.save_quality_report(df_with_sentiment, quality_report_path)
    
    quality_score = quality_metrics.get_data_quality_score(df_with_sentiment)
    print(f"📊 Data quality score: {quality_score:.1f}/100")
    
    print(f"✅ Sentiment analysis completed. Results saved to {output_path}")
    return df_with_sentiment


def analyze_absa(config: ConfigManager, df) -> None:
    """Perform aspect-based sentiment analysis."""
    print("🎯 Performing aspect-based sentiment analysis...")
    
    # Load sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(config)
    sentiment_analyzer.load_model()
    
    # Create ABSA analyzer
    absa_analyzer = ABSAAnalyzer(config, sentiment_analyzer)
    
    # Perform ABSA
    df_absa = absa_analyzer.analyze_dataframe(df)
    
    # Save results
    output_path = config.get_data_path('absa_results.csv')
    absa_analyzer.save_results(df_absa, output_path)
    
    # Generate and save aspect summary
    aspect_summary = absa_analyzer.get_aspect_summary(df_absa)
    summary_path = config.get_data_path('aspect_summary.csv')
    aspect_summary.to_csv(summary_path, index=False)
    
    # Generate ABSA quality report
    quality_metrics = QualityMetrics(config)
    absa_quality_path = config.get_data_path('absa_quality_report.txt')
    quality_metrics.save_quality_report(df_absa, absa_quality_path)
    
    absa_quality_score = quality_metrics.get_data_quality_score(df_absa)
    print(f"📊 ABSA quality score: {absa_quality_score:.1f}/100")
    
    print(f"✅ ABSA analysis completed. Results saved to {output_path}")
    print(f"📊 Aspect summary saved to {summary_path}")
    return df_absa


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Board Game NLP Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --collect-data                    # Collect board game data
  python main.py --train-model                     # Train sentiment model
  python main.py --analyze-sentiment               # Analyze sentiment
  python main.py --analyze-absa                    # Perform ABSA
  python main.py --full-pipeline                   # Run complete pipeline
        """
    )
    
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--collect-data', action='store_true',
                       help='Collect board game data from BGG API')
    parser.add_argument('--train-model', action='store_true',
                       help='Train sentiment analysis model')
    parser.add_argument('--analyze-sentiment', action='store_true',
                       help='Perform sentiment analysis on comments')
    parser.add_argument('--analyze-absa', action='store_true',
                       help='Perform aspect-based sentiment analysis')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run the complete analysis pipeline')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = ConfigManager(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Setup directories
    setup_directories(config)
    
    # Run selected operations
    try:
        if args.full_pipeline or args.collect_data:
            df = collect_data(config)
        
        if args.full_pipeline or args.train_model:
            train_model(config)
        
        if args.full_pipeline or args.analyze_sentiment:
            if 'df' not in locals():
                # Load existing data if not collected
                import pandas as pd
                df = pd.read_csv(config.get_data_path('english_boardgames_comments.csv'))
            analyze_sentiment(config, df)
        
        if args.full_pipeline or args.analyze_absa:
            if 'df' not in locals():
                # Load existing data if not collected
                import pandas as pd
                df = pd.read_csv(config.get_data_path('english_boardgames_comments.csv'))
            analyze_absa(config, df)
        
        if not any([args.collect_data, args.train_model, args.analyze_sentiment, 
                   args.analyze_absa, args.full_pipeline]):
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        config.logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    
    print("🎉 Analysis pipeline completed successfully!")


if __name__ == "__main__":
    main()
