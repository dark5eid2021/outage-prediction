# Production Log Analyzer and Outage Predictor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a proof of concept for an intelligent system that analyzes production logs and predicts whether recent commits to source control will cause outages or failures. The system uses machine learning to combine commit metadata, code changes, and real-time log analysis to provide early warning of potential production issues.

## 🏗️ Architecture

![Outage Prediction System Architecture](./docs/architecture-diagram.svg)

The system consists of several key components working together to provide intelligent outage prediction:
- **Data Sources**: Git repositories, log files, and historical outage data
- **Data Processing**: Git analyzer, log parser, and feature extractor
- **Machine Learning Engine**: Training module, prediction engine, and model storage
- **Database**: SQLite/PostgreSQL for storing predictions, outcomes, and feedback
- **Outputs**: Risk reports, monitoring integrations, and API interfaces
- **Feedback Loop**: Continuous learning from actual outcomes

## 🚀 Features

- **Multi-format Log Parsing**: Supports JSON, standard application logs, nginx, and Apache formats
- **Git Integration**: Analyzes recent commits including size, files changed, and timing
- **Machine Learning Prediction**: Uses Random Forest classifier to predict outage probability
- **Real-time Anomaly Detection**: Identifies unusual patterns in log data
- **Risk Assessment**: Provides risk levels (LOW, MEDIUM, HIGH, CRITICAL) with confidence scores
- **Actionable Recommendations**: Suggests specific actions based on detected risk factors
- **Historical Feedback**: Learns from actual outcomes to improve prediction accuracy
- **Database Storage**: Tracks predictions and outcomes for continuous improvement

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Git repository access
- Read access to production log files

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
```

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/outage-predictor.git
   cd outage-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the database:**
   ```bash
   # The database will be automatically created on first run
   python log_analyzer.py --repo-path /path/to/repo --log-files /path/to/logs/*.log --train
   ```

## 🎯 Quick Start

### Basic Usage

```bash
# Analyze recent commits and predict outage risk
python log_analyzer.py \
  --repo-path /path/to/your/git/repo \
  --log-files /var/log/application.log /var/log/nginx/access.log \
  --hours-back 24
```

### Train the Model

```bash
# Train the prediction model (do this first)
python log_analyzer.py \
  --repo-path /path/to/your/git/repo \
  --log-files /var/log/*.log \
  --train
```

### Example Output

```
================================================================================
OUTAGE RISK PREDICTION REPORT
================================================================================
Analysis time: 2025-06-19 14:30:25
Repository: /path/to/your/repo
Time window: 24 hours
Total commits analyzed: 3
================================================================================

COMMIT #1
Hash: a1b2c3d4e5f6
Risk Level: HIGH
Risk Score: 0.782
Confidence: 0.782
Contributing Factors:
  • Large commit size
  • High current error rate
  • After-hours deployment
Recommended Actions:
  → Consider rolling back this deployment
  → Increase monitoring and alerting
  → Have on-call team ready
  → Wait for current issues to be resolved before deploying
------------------------------------------------------------
```

## 📊 How It Works

### 1. Log Analysis
The system parses various log formats and extracts:
- Error rates and patterns
- Service health metrics
- Anomaly detection scores
- Trend analysis over time

### 2. Commit Analysis
For each recent commit, it analyzes:
- **Size metrics**: Lines added/deleted, files changed
- **Risk factors**: File types, commit timing, message keywords
- **Change patterns**: Diff analysis and impact assessment

### 3. Feature Engineering
Combines log and commit data into ML features:
- `commit_size_score`: Normalized size of changes
- `high_risk_files`: Count of critical files modified
- `current_error_rate`: Recent error rate from logs
- `anomaly_score`: Statistical anomaly detection
- `weekend_commit`: Boolean for weekend deployments
- `after_hours`: Boolean for after-hours deployments

### 4. Risk Prediction
Uses a Random Forest classifier to predict:
- **Risk probability**: 0.0 to 1.0 scale
- **Risk level**: LOW, MEDIUM, HIGH, CRITICAL
- **Confidence**: Model confidence in prediction
- **Contributing factors**: Specific risk elements identified
- **Recommendations**: Actionable steps to mitigate risk

## 🛠️ Advanced Configuration

### Custom Log Patterns

Extend the `LogParser` class to support custom log formats:

```python
parser = LogParser()
parser.PATTERNS['custom'] = r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}) (?P<level>\w+) (?P<message>.*)'
```

### Model Tuning

Adjust the Random Forest parameters:

```python
predictor = OutagePredictor(repo_path)
predictor.model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

### Custom Risk Factors

Add domain-specific risk factors by extending the `extract_features` method:

```python
def extract_custom_features(self, commit, log_analysis):
    features = self.extract_features(commit, log_analysis)
    
    # Add custom features
    features['database_changes'] = sum(1 for f in commit.files_changed 
                                     if 'migration' in f or '.sql' in f)
    features['config_changes'] = sum(1 for f in commit.files_changed 
                                   if 'config' in f or '.yml' in f)
    
    return features
```

## 📈 Model Training

### Using Historical Data

The system learns from historical commit and outage data. To improve accuracy:

1. **Collect historical data**: Export commit history and known outage incidents
2. **Label training data**: Mark commits that caused outages (1) vs. safe commits (0)
3. **Retrain the model**: Use the collected data to train a more accurate model

```python
# Example training data format
training_data = [
    ({'lines_changed': 150, 'error_rate': 0.02, ...}, 0),  # No outage
    ({'lines_changed': 800, 'error_rate': 0.08, ...}, 1),  # Caused outage
]
```

### Continuous Learning

Add feedback for continuous improvement:

```python
# After deployment, record actual outcomes
predictor.add_feedback('commit_hash', 'NO_OUTAGE')  # or 'OUTAGE_OCCURRED'
```

## 🔍 Monitoring and Alerting

### Integration with Monitoring Systems

The predictor can be integrated with existing monitoring infrastructure:

```python
# Example Slack notification for high-risk commits
def send_alert(prediction):
    if prediction.risk_level in ['HIGH', 'CRITICAL']:
        message = f"⚠️ High risk deployment detected!\n"
        message += f"Commit: {prediction.commit_hash}\n"
        message += f"Risk: {prediction.risk_level} ({prediction.risk_score:.2f})\n"
        message += f"Actions: {', '.join(prediction.recommended_actions)}"
        
        # Send to Slack, PagerDuty, etc.
        send_slack_message(message)
```

### Automated Deployment Gates

Use predictions to gate deployments:

```python
def should_deploy(commit_hash, max_risk_score=0.6):
    prediction = predictor.predict_single_commit(commit_hash)
    return prediction.risk_score < max_risk_score
```

## 📝 API Reference

### OutagePredictor Class

```python
class OutagePredictor:
    def __init__(self, repo_path: str, model_path: str = 'outage_model.pkl')
    def predict_outage_risk(self, log_files: List[str], hours_back: int = 24) -> List[OutagePrediction]
    def train_model(self, training_data_path: str = None)
    def add_feedback(self, commit_hash: str, actual_outcome: str)
```

### OutagePrediction Class

```python
@dataclass
class OutagePrediction:
    commit_hash: str
    risk_score: float          # 0.0 to 1.0
    risk_level: str           # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float         # Model confidence
    contributing_factors: List[str]
    recommended_actions: List[str]
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Test with sample data:

```bash
python log_analyzer.py \
  --repo-path ./sample_repo \
  --log-files ./sample_logs/app.log \
  --hours-back 1
```

## 🔒 Security Considerations

- **Log Sanitization**: Ensure logs don't contain sensitive information
- **Access Control**: Restrict access to production logs and git repositories
- **Data Retention**: Implement appropriate data retention policies
- **Audit Trail**: Log all predictions and access for compliance

## 📈 Performance Optimization

### For Large Repositories
- Use `--hours-back` to limit analysis window
- Consider sampling large log files
- Implement log rotation and archiving

### For High-Volume Logs
- Parse logs in chunks
- Use streaming analysis for real-time processing
- Consider distributed processing for very large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Run linting
flake8 log_analyzer.py
black log_analyzer.py
```

## 📊 Metrics and Monitoring

The system tracks several key metrics:

- **Prediction Accuracy**: How often predictions match actual outcomes
- **False Positive Rate**: Predictions of outages that don't occur
- **False Negative Rate**: Missed outages that weren't predicted
- **Response Time**: Time to analyze and generate predictions
- **Coverage**: Percentage of deployments analyzed

## 🐛 Troubleshooting

### Common Issues

1. **"No recent commits found"**
   - Check git repository path
   - Verify git is accessible
   - Ensure commits exist in the specified time window

2. **"Error parsing log file"**
   - Check log file permissions
   - Verify log format is supported
   - Check for file encoding issues

3. **"Model file not found"**
   - Run with `--train` flag first
   - Check model file permissions
   - Verify sufficient training data

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn team for the machine learning library
- Git community for version control insights
- Production engineering best practices from various sources

## 📞 Support

- Create an issue for bug reports
- Check existing issues for known problems
- Join our community discussions

---

**Note**: This system provides predictions based on historical patterns and current log analysis. Always use human judgment and established deployment procedures alongside these predictions.