# Board Game NLP Analysis

A comprehensive Natural Language Processing toolkit for analyzing board game reviews and comments using sentiment analysis and aspect-based sentiment analysis (ABSA).

## 🎯 Features

- **Data Collection**: Automated scraping of board game comments from BoardGameGeek API
- **Sentiment Analysis**: Binary classification (positive/negative) using fine-tuned ALBERT model
- **Aspect-Based Sentiment Analysis (ABSA)**: Analyze sentiment for specific game aspects:
  - **Luck/Chance**: randomness, luck, chance
  - **Bookkeeping**: recording, rulebook, manual, tracking
  - **Downtime**: waiting, idle time, turn management
  - **Interaction**: player interaction, influence, impact
  - **Bash the Leader**: targeting leading players
  - **Complexity**: rules complexity, difficulty
- **Model Training**: Fine-tune ALBERT on IMDB dataset for sentiment classification
- **Comprehensive Results**: Detailed analysis with confidence scores and aspect summaries

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd boardgame
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**:
   ```bash
   python main.py --full-pipeline
   ```

### Individual Components

```bash
# Collect board game data
python main.py --collect-data

# Train sentiment analysis model
python main.py --train-model

# Analyze sentiment
python main.py --analyze-sentiment

# Perform aspect-based sentiment analysis
python main.py --analyze-absa
```

## 📁 Project Structure

```
boardgame/
├── boardgame_nlp/           # Main package
│   ├── __init__.py
│   ├── data_collector.py    # BGG API data collection
│   ├── sentiment_analyzer.py # Sentiment analysis
│   ├── absa_analyzer.py     # Aspect-based sentiment analysis
│   ├── model_trainer.py     # Model training
│   └── utils.py             # Utility functions
├── BGG/                     # Original notebooks
├── IMDB/                    # Model training notebooks
├── Model/                   # Trained models
├── Results/                 # Analysis results
├── data/                    # Data files
├── config.yaml             # Configuration
├── main.py                 # Main execution script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## ⚙️ Configuration

The project uses `config.yaml` for configuration. Key settings:

```yaml
# Data paths
data:
  base_path: "./data"
  comments_file: "boardgames_comments.csv"
  english_comments_file: "english_boardgames_comments.csv"

# Model settings
model:
  name: "albert-base-v2"
  max_length: 512
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 3

# API settings
api:
  base_url: "https://api.geekdo.com/xmlapi2/thing"
  rate_limit_delay: 1.5
  max_retries: 3
```

## 🔧 Usage

### Data Collection

The data collector automatically:
- Downloads top 10 board games from BGG rankings
- Scrapes comments for each game
- Filters for English comments
- Cleans and preprocesses text

### Model Training

The model trainer:
- Loads IMDB dataset for training
- Fine-tunes ALBERT-base-v2 for binary classification
- Achieves ~93% accuracy on test set
- Saves trained model for inference

### Sentiment Analysis

Performs binary sentiment classification:
- Positive/Negative classification
- Confidence scores
- Batch processing support

### Aspect-Based Sentiment Analysis

Analyzes sentiment for specific game aspects:
- Keyword-based aspect detection
- Aspect-specific sentiment analysis
- Comprehensive aspect summaries

## 📊 Results

The analysis produces several output files:

- `sentiment_analysis_results.csv`: General sentiment analysis results
- `absa_results.csv`: Aspect-based sentiment analysis results
- `aspect_summary.csv`: Summary statistics for each aspect

### Example Output

```csv
boardgame_id,cleaned_comment,sentiment,confidence,detected_aspects,aspect_sentiments
224517,"Great game with lots of interaction",positive,0.95,"['interaction']","{'interaction': 'positive'}"
```

## 🛠️ Development

### Adding New Aspects

To add new aspects for ABSA, update the `aspects` section in `config.yaml`:

```yaml
aspects:
  new_aspect: ["keyword1", "keyword2", "keyword3"]
```

### Custom Model Training

To train on different data:

```python
from boardgame_nlp import ModelTrainer
from boardgame_nlp.utils import ConfigManager

config = ConfigManager()
trainer = ModelTrainer(config)
trainer.setup_model()
# Custom training logic here
```

## 📈 Performance

- **Model Accuracy**: ~93.4% on IMDB test set
- **Processing Speed**: ~100 comments/second on CPU
- **Memory Usage**: ~2GB for model inference
- **API Rate Limiting**: Respects BGG API limits (1.5s delay)

## 🔍 Troubleshooting

### Common Issues

1. **API Rate Limiting**: Increase `rate_limit_delay` in config
2. **Memory Issues**: Reduce `batch_size` in config
3. **Model Loading**: Ensure model files are in correct directory

### Logs

Check `boardgame_analysis.log` for detailed execution logs.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the logs in `boardgame_analysis.log`
- Review the configuration in `config.yaml`

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time analysis dashboard
- [ ] Advanced aspect detection using NER
- [ ] Model ensemble methods
- [ ] Interactive visualization tools
