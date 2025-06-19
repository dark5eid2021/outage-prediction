#!/usr/bin/env python3
"""
Production Log Analyzer and Outage Predictor
============================================

This system analyzes production logs and predicts whether recent commits
might cause outages or failures based on historical patterns and log anomalies.
"""

import os
import re
import json
import logging
import hashlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import sqlite3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Represents a single log entry with metadata"""
    timestamp: datetime
    level: str
    message: str
    service: str
    thread_id: Optional[str] = None
    request_id: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None


@dataclass
class CommitInfo:
    """Represents commit information from source control"""
    commit_hash: str
    author: str
    timestamp: datetime
    message: str
    files_changed: List[str]
    lines_added: int
    lines_deleted: int
    diff_summary: str


@dataclass
class OutagePrediction:
    """Represents a prediction result"""
    commit_hash: str
    risk_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    contributing_factors: List[str]
    recommended_actions: List[str]


class LogParser:
    """Parses various log formats and extracts structured information"""
    
    # Common log patterns
    PATTERNS = {
        'standard': r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} \[(?P<thread>[^\]]+)\] (?P<level>\w+) +(?P<logger>[^\s]+) - (?P<message>.*)',
        'nginx': r'(?P<ip>\S+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"',
        'apache': r'(?P<ip>\S+) - - \[(?P<timestamp>[^\]]+)\] "(?P<request>[^"]*)" (?P<status>\d+) (?P<size>\S+)',
        'json': r'^\{.*\}$'
    }
    
    def __init__(self):
        self.compiled_patterns = {name: re.compile(pattern) 
                                for name, pattern in self.PATTERNS.items()}
    
    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry object"""
        line = line.strip()
        if not line:
            return None
            
        # Try JSON format first
        if line.startswith('{') and line.endswith('}'):
            try:
                data = json.loads(line)
                return LogEntry(
                    timestamp=datetime.fromisoformat(data.get('timestamp', '1970-01-01T00:00:00')),
                    level=data.get('level', 'INFO'),
                    message=data.get('message', ''),
                    service=data.get('service', 'unknown'),
                    thread_id=data.get('thread_id'),
                    request_id=data.get('request_id'),
                    error_code=data.get('error_code'),
                    stack_trace=data.get('stack_trace')
                )
            except json.JSONDecodeError:
                pass
        
        # Try standard patterns
        for pattern_name, pattern in self.compiled_patterns.items():
            if pattern_name == 'json':
                continue
                
            match = pattern.match(line)
            if match:
                groups = match.groupdict()
                
                # Parse timestamp
                timestamp_str = groups.get('timestamp', '')
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    timestamp = datetime.now()
                
                return LogEntry(
                    timestamp=timestamp,
                    level=groups.get('level', 'INFO'),
                    message=groups.get('message', line),
                    service=groups.get('logger', 'unknown'),
                    thread_id=groups.get('thread')
                )
        
        # Fallback: create entry with raw line
        return LogEntry(
            timestamp=datetime.now(),
            level='INFO',
            message=line,
            service='unknown'
        )
    
    def parse_log_file(self, filepath: str) -> List[LogEntry]:
        """Parse an entire log file"""
        entries = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = self.parse_log_line(line)
                        if entry:
                            entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num} in {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error reading log file {filepath}: {e}")
        
        return entries


class GitAnalyzer:
    """Analyzes Git repository for commit information"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
    
    def get_recent_commits(self, hours: int = 24) -> List[CommitInfo]:
        """Get commits from the last N hours"""
        since_time = datetime.now() - timedelta(hours=hours)
        since_str = since_time.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # Get commit hashes and basic info
            cmd = ['git', 'log', '--since', since_str, '--pretty=format:%H|%an|%ad|%s', '--date=iso']
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Git log failed: {result.stderr}")
                return []
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split('|', 3)
                if len(parts) < 4:
                    continue
                
                commit_hash, author, timestamp_str, message = parts
                timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                
                # Get detailed commit info
                commit_info = self._get_commit_details(commit_hash)
                commits.append(CommitInfo(
                    commit_hash=commit_hash,
                    author=author,
                    timestamp=timestamp,
                    message=message,
                    **commit_info
                ))
            
            return commits
            
        except Exception as e:
            logger.error(f"Error getting recent commits: {e}")
            return []
    
    def _get_commit_details(self, commit_hash: str) -> Dict[str, Any]:
        """Get detailed information about a specific commit"""
        try:
            # Get files changed
            cmd = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            files_changed = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get line changes
            cmd = ['git', 'show', '--stat', '--format=', commit_hash]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            lines_added, lines_deleted = self._parse_stat_output(result.stdout)
            
            # Get diff summary
            cmd = ['git', 'show', '--name-status', commit_hash]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            diff_summary = result.stdout
            
            return {
                'files_changed': files_changed,
                'lines_added': lines_added,
                'lines_deleted': lines_deleted,
                'diff_summary': diff_summary
            }
            
        except Exception as e:
            logger.error(f"Error getting commit details for {commit_hash}: {e}")
            return {
                'files_changed': [],
                'lines_added': 0,
                'lines_deleted': 0,
                'diff_summary': ''
            }
    
    def _parse_stat_output(self, stat_output: str) -> Tuple[int, int]:
        """Parse git stat output to extract lines added/deleted"""
        lines_added = lines_deleted = 0
        
        for line in stat_output.split('\n'):
            if 'insertions(+)' in line or 'deletions(-)' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'insertions(+)' in part and i > 0:
                        lines_added = int(parts[i-1])
                    elif 'deletions(-)' in part and i > 0:
                        lines_deleted = int(parts[i-1])
        
        return lines_added, lines_deleted


class LogAnomalyDetector:
    """Detects anomalies in log patterns that might indicate issues"""
    
    def __init__(self):
        self.error_patterns = [
            r'error|exception|failed|failure|fatal|critical',
            r'timeout|connection.*reset|connection.*refused',
            r'out of memory|memory.*exceeded|heap.*space',
            r'deadlock|lock.*timeout|database.*error',
            r'500|502|503|504|internal.*server.*error'
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                for pattern in self.error_patterns]
    
    def analyze_logs(self, log_entries: List[LogEntry], time_window_minutes: int = 30) -> Dict[str, Any]:
        """Analyze logs for anomalies and patterns"""
        if not log_entries:
            return {}
        
        # Sort by timestamp
        log_entries.sort(key=lambda x: x.timestamp)
        
        # Time window analysis
        window_start = log_entries[-1].timestamp - timedelta(minutes=time_window_minutes)
        recent_logs = [entry for entry in log_entries if entry.timestamp >= window_start]
        
        analysis = {
            'total_entries': len(log_entries),
            'recent_entries': len(recent_logs),
            'error_rate': self._calculate_error_rate(recent_logs),
            'anomaly_score': self._calculate_anomaly_score(recent_logs),
            'error_patterns': self._detect_error_patterns(recent_logs),
            'service_health': self._analyze_service_health(recent_logs),
            'trend_analysis': self._analyze_trends(log_entries)
        }
        
        return analysis
    
    def _calculate_error_rate(self, entries: List[LogEntry]) -> float:
        """Calculate the error rate in log entries"""
        if not entries:
            return 0.0
        
        error_count = sum(1 for entry in entries 
                         if entry.level.upper() in ['ERROR', 'FATAL', 'CRITICAL'])
        return error_count / len(entries)
    
    def _calculate_anomaly_score(self, entries: List[LogEntry]) -> float:
        """Calculate overall anomaly score based on various factors"""
        if not entries:
            return 0.0
        
        score = 0.0
        
        # Error rate contribution
        error_rate = self._calculate_error_rate(entries)
        score += error_rate * 0.4
        
        # Pattern matching contribution
        pattern_matches = sum(1 for entry in entries 
                            for pattern in self.compiled_patterns 
                            if pattern.search(entry.message))
        pattern_score = min(pattern_matches / len(entries), 1.0)
        score += pattern_score * 0.3
        
        # Service diversity (many services having issues is bad)
        services = set(entry.service for entry in entries)
        service_score = len(services) / max(len(services), 10)  # Normalize
        score += service_score * 0.3
        
        return min(score, 1.0)
    
    def _detect_error_patterns(self, entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect specific error patterns in logs"""
        patterns = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            matches = [entry for entry in entries if pattern.search(entry.message)]
            if matches:
                patterns.append({
                    'pattern_id': i,
                    'pattern': self.error_patterns[i],
                    'count': len(matches),
                    'rate': len(matches) / len(entries) if entries else 0,
                    'sample_messages': [entry.message for entry in matches[:3]]
                })
        
        return patterns
    
    def _analyze_service_health(self, entries: List[LogEntry]) -> Dict[str, Dict[str, Any]]:
        """Analyze health of individual services"""
        service_stats = defaultdict(lambda: {'total': 0, 'errors': 0, 'messages': []})
        
        for entry in entries:
            service_stats[entry.service]['total'] += 1
            if entry.level.upper() in ['ERROR', 'FATAL', 'CRITICAL']:
                service_stats[entry.service]['errors'] += 1
            service_stats[entry.service]['messages'].append(entry.message)
        
        health_report = {}
        for service, stats in service_stats.items():
            error_rate = stats['errors'] / stats['total'] if stats['total'] > 0 else 0
            health_report[service] = {
                'total_logs': stats['total'],
                'error_count': stats['errors'],
                'error_rate': error_rate,
                'health_status': 'CRITICAL' if error_rate > 0.1 else 'WARNING' if error_rate > 0.05 else 'HEALTHY'
            }
        
        return health_report
    
    def _analyze_trends(self, entries: List[LogEntry]) -> Dict[str, Any]:
        """Analyze trends in log data over time"""
        if len(entries) < 10:
            return {'insufficient_data': True}
        
        # Group by hour
        hourly_counts = defaultdict(int)
        hourly_errors = defaultdict(int)
        
        for entry in entries:
            hour_key = entry.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
            if entry.level.upper() in ['ERROR', 'FATAL', 'CRITICAL']:
                hourly_errors[hour_key] += 1
        
        # Calculate trends
        hours = sorted(hourly_counts.keys())
        if len(hours) >= 2:
            recent_error_rate = hourly_errors[hours[-1]] / max(hourly_counts[hours[-1]], 1)
            prev_error_rate = hourly_errors[hours[-2]] / max(hourly_counts[hours[-2]], 1)
            error_trend = recent_error_rate - prev_error_rate
        else:
            error_trend = 0
        
        return {
            'error_trend': error_trend,
            'trend_direction': 'INCREASING' if error_trend > 0.01 else 'DECREASING' if error_trend < -0.01 else 'STABLE'
        }


class OutagePredictor:
    """Main class that combines log analysis and commit analysis to predict outages"""
    
    def __init__(self, repo_path: str, model_path: str = 'outage_model.pkl'):
        self.repo_path = repo_path
        self.model_path = model_path
        self.log_parser = LogParser()
        self.git_analyzer = GitAnalyzer(repo_path)
        self.anomaly_detector = LogAnomalyDetector()
        self.model = None
        self.vectorizer = None
        self.db_path = 'outage_predictions.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing predictions and feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_hash TEXT UNIQUE,
                timestamp DATETIME,
                risk_score REAL,
                risk_level TEXT,
                confidence REAL,
                prediction_data TEXT,
                actual_outcome TEXT,
                feedback_timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_hash TEXT,
                features TEXT,
                label INTEGER,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_features(self, commit: CommitInfo, log_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for ML model"""
        features = {}
        
        # Commit-based features
        features['lines_changed'] = commit.lines_added + commit.lines_deleted
        features['files_changed'] = len(commit.files_changed)
        features['commit_size_score'] = min((features['lines_changed'] / 1000), 1.0)
        
        # File type risk scores
        high_risk_extensions = ['.sql', '.py', '.java', '.js', '.config', '.yaml', '.yml']
        features['high_risk_files'] = sum(1 for f in commit.files_changed 
                                        if any(f.endswith(ext) for ext in high_risk_extensions))
        
        # Commit message analysis
        risky_keywords = ['hotfix', 'urgent', 'critical', 'emergency', 'quick', 'temp', 'hack']
        features['risky_commit_message'] = sum(1 for keyword in risky_keywords 
                                             if keyword.lower() in commit.message.lower())
        
        # Time-based features
        features['weekend_commit'] = 1 if commit.timestamp.weekday() >= 5 else 0
        features['after_hours'] = 1 if commit.timestamp.hour < 8 or commit.timestamp.hour > 18 else 0
        
        # Log-based features
        if log_analysis:
            features['current_error_rate'] = log_analysis.get('error_rate', 0)
            features['anomaly_score'] = log_analysis.get('anomaly_score', 0)
            features['error_pattern_count'] = len(log_analysis.get('error_patterns', []))
            
            # Service health
            service_health = log_analysis.get('service_health', {})
            critical_services = sum(1 for health in service_health.values() 
                                  if health.get('health_status') == 'CRITICAL')
            features['critical_services'] = critical_services
            
            # Trend analysis
            trend = log_analysis.get('trend_analysis', {})
            features['error_trend_increasing'] = 1 if trend.get('trend_direction') == 'INCREASING' else 0
        else:
            # Default values when no log analysis available
            features.update({
                'current_error_rate': 0,
                'anomaly_score': 0,
                'error_pattern_count': 0,
                'critical_services': 0,
                'error_trend_increasing': 0
            })
        
        return features
    
    def train_model(self, training_data_path: str = None):
        """Train the ML model using historical data"""
        logger.info("Training outage prediction model...")
        
        # In a real implementation, you would load historical data
        # For this example, we'll create some synthetic training data
        training_features = []
        training_labels = []
        
        # This would be replaced with actual historical data
        # Format: features dict and label (0 = no outage, 1 = outage)
        synthetic_data = self._generate_synthetic_training_data()
        
        for features, label in synthetic_data:
            training_features.append(list(features.values()))
            training_labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(training_features)
        y = np.array(training_labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        logger.info(f"Model accuracy: {self.model.score(X_test, y_test):.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def _generate_synthetic_training_data(self) -> List[Tuple[Dict[str, float], int]]:
        """Generate synthetic training data for demonstration"""
        # This would be replaced with actual historical data loading
        data = []
        
        # Generate some synthetic examples
        for i in range(1000):
            features = {
                'lines_changed': np.random.randint(1, 1000),
                'files_changed': np.random.randint(1, 50),
                'commit_size_score': np.random.random(),
                'high_risk_files': np.random.randint(0, 10),
                'risky_commit_message': np.random.randint(0, 3),
                'weekend_commit': np.random.randint(0, 2),
                'after_hours': np.random.randint(0, 2),
                'current_error_rate': np.random.random() * 0.1,
                'anomaly_score': np.random.random(),
                'error_pattern_count': np.random.randint(0, 5),
                'critical_services': np.random.randint(0, 3),
                'error_trend_increasing': np.random.randint(0, 2)
            }
            
            # Create label based on risk factors (synthetic logic)
            risk_score = (
                features['commit_size_score'] * 0.3 +
                features['current_error_rate'] * 0.3 +
                features['anomaly_score'] * 0.2 +
                features['risky_commit_message'] * 0.1 +
                features['weekend_commit'] * 0.05 +
                features['after_hours'] * 0.05
            )
            
            label = 1 if risk_score > 0.6 else 0
            data.append((features, label))
        
        return data
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            logger.warning(f"Model file {self.model_path} not found. Training new model...")
            self.train_model()
    
    def predict_outage_risk(self, log_files: List[str], hours_back: int = 24) -> List[OutagePrediction]:
        """Main method to predict outage risk for recent commits"""
        logger.info("Starting outage risk prediction...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Get recent commits
        commits = self.git_analyzer.get_recent_commits(hours_back)
        if not commits:
            logger.info("No recent commits found")
            return []
        
        # Parse log files
        all_log_entries = []
        for log_file in log_files:
            if os.path.exists(log_file):
                entries = self.log_parser.parse_log_file(log_file)
                all_log_entries.extend(entries)
                logger.info(f"Parsed {len(entries)} entries from {log_file}")
        
        # Analyze logs
        log_analysis = self.anomaly_detector.analyze_logs(all_log_entries)
        
        # Generate predictions for each commit
        predictions = []
        for commit in commits:
            try:
                prediction = self._predict_single_commit(commit, log_analysis)
                predictions.append(prediction)
                
                # Store prediction in database
                self._store_prediction(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting for commit {commit.commit_hash}: {e}")
        
        # Sort by risk score (highest first)
        predictions.sort(key=lambda x: x.risk_score, reverse=True)
        
        return predictions
    
    def _predict_single_commit(self, commit: CommitInfo, log_analysis: Dict[str, Any]) -> OutagePrediction:
        """Predict outage risk for a single commit"""
        # Extract features
        features = self.extract_features(commit, log_analysis)
        
        # Prepare features for model
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        
        # Get prediction
        risk_probability = self.model.predict_proba(feature_vector)[0][1]  # Probability of outage
        confidence = max(risk_probability, 1 - risk_probability)
        
        # Determine risk level
        if risk_probability >= 0.8:
            risk_level = "CRITICAL"
        elif risk_probability >= 0.6:
            risk_level = "HIGH"
        elif risk_probability >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Identify contributing factors
        contributing_factors = []
        if features['commit_size_score'] > 0.5:
            contributing_factors.append("Large commit size")
        if features['risky_commit_message'] > 0:
            contributing_factors.append("Risky keywords in commit message")
        if features['current_error_rate'] > 0.05:
            contributing_factors.append("High current error rate")
        if features['anomaly_score'] > 0.3:
            contributing_factors.append("Log anomalies detected")
        if features['weekend_commit']:
            contributing_factors.append("Weekend deployment")
        if features['after_hours']:
            contributing_factors.append("After-hours deployment")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_level, contributing_factors)
        
        return OutagePrediction(
            commit_hash=commit.commit_hash,
            risk_score=risk_probability,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            recommended_actions=recommendations
        )
    
    def _generate_recommendations(self, risk_level: str, factors: List[str]) -> List[str]:
        """Generate recommendations based on risk level and factors"""
        recommendations = []
        
        if risk_level in ["CRITICAL", "HIGH"]:
            recommendations.append("Consider rolling back this deployment")
            recommendations.append("Increase monitoring and alerting")
            recommendations.append("Have on-call team ready")
        
        if "Large commit size" in factors:
            recommendations.append("Consider breaking large changes into smaller commits")
        
        if "High current error rate" in factors:
            recommendations.append("Wait for current issues to be resolved before deploying")
        
        if "Weekend deployment" in factors or "After-hours deployment" in factors:
            recommendations.append("Consider deploying during business hours for better support")
        
        if "Log anomalies detected" in factors:
            recommendations.append("Investigate current system anomalies before proceeding")
        
        if not recommendations:
            recommendations.append("Monitor deployment closely")
            recommendations.append("Ensure rollback plan is ready")
        
        return recommendations
    
    def _store_prediction(self, prediction: OutagePrediction):
        """Store prediction in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO predictions 
            (commit_hash, timestamp, risk_score, risk_level, confidence, prediction_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            prediction.commit_hash,
            datetime.now(),
            prediction.risk_score,
            prediction.risk_level,
            prediction.confidence,
            json.dumps(asdict(prediction))
        ))
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, commit_hash: str, actual_outcome: str):
        """Add feedback about actual outcome for model improvement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_outcome = ?, feedback_timestamp = ?
            WHERE commit_hash = ?
        ''', (actual_outcome, datetime.now(), commit_hash))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback added for commit {commit_hash}: {actual_outcome}")


def main():
    """Main function to demonstrate usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict outage risk from commits and logs')
    parser.add_argument('--repo-path', required=True, help='Path to git repository')
    parser.add_argument('--log-files', nargs='+', required=True, help='Log files to analyze')
    parser.add_argument('--hours-back', type=int, default=24, help='Hours back to analyze commits')
    parser.add_argument('--train', action='store_true', help='Train the model')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = OutagePredictor(args.repo_path)
    
    # Train model if requested
    if args.train:
        predictor.train_model()
    
    # Generate predictions
    predictions = predictor.predict_outage_risk(args.log_files, args.hours_back)
    
    # Display results
    if not predictions:
        print("No recent commits found or no risk detected.")
        return
    
    print(f"\n{'='*80}")
    print(f"OUTAGE RISK PREDICTION REPORT")
    print(f"{'='*80}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Repository: {args.repo_path}")
    print(f"Time window: {args.hours_back} hours")
    print(f"Total commits analyzed: {len(predictions)}")
    print(f"{'='*80}\n")
    
    for i, prediction in enumerate(predictions, 1):
        print(f"COMMIT #{i}")
        print(f"Hash: {prediction.commit_hash}")
        print(f"Risk Level: {prediction.risk_level}")
        print(f"Risk Score: {prediction.risk_score:.3f}")
        print(f"Confidence: {prediction.confidence:.3f}")
        
        if prediction.contributing_factors:
            print("Contributing Factors:")
            for factor in prediction.contributing_factors:
                print(f"  • {factor}")
        
        if prediction.recommended_actions:
            print("Recommended Actions:")
            for action in prediction.recommended_actions:
                print(f"  → {action}")
        
        print("-" * 60)


if __name__ == "__main__":
    main()