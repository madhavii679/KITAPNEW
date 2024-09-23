import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from scipy.stats import t
import io
import base64
import warnings
import seaborn as sns
import logging
import json
import yaml
from typing import Dict, List, Tuple, Any
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_insights.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

class DataValidator:
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> bool:
        required_columns = [
            'timestamp', 'status', 'duration', 'cpu_usage', 
            'memory_usage', 'response_time', 'Test_CASE_ID', 
            'environment', 'lines_added', 'lines_removed'
        ]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        return True

class DataProcessor:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = df['timestamp'].dt.date
        df['total_lines_changed'] = df['lines_added'] + df['lines_removed']
        return df

class AnalysisBase:
    def __init__(self, df: pd.DataFrame, config: ConfigManager):
        self.df = df
        self.config = config

    def plot_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)
        return img_str

class OverallInsightsAnalysis(AnalysisBase):
    def analyze(self) -> Tuple[str, List[str], List[str], pd.DataFrame]:
        total_executions = len(self.df)
        pass_rate = (self.df['status'] == 'Passed').mean() * 100
        avg_duration = self.df['duration'].mean()
        max_duration = self.df['duration'].max()
        min_duration = self.df['duration'].min()

        fig, ax = plt.subplots(figsize=(
            self.config.get('visualization', 'figure_size', 'width', default=10), 
            self.config.get('visualization', 'figure_size', 'height', default=6)
        ))

        colors = [
            self.config.get('visualization', 'colors', 'primary', default='green') 
            if status == 'Passed' else 
            self.config.get('visualization', 'colors', 'secondary', default='red') 
            for status in self.df['status'].value_counts().index
        ]

        self.df['status'].value_counts().plot(kind='bar', ax=ax, color=colors)
        
        ax.set_title('Overall Test Status Distribution')
        ax.set_ylabel('Number of Test Case Executions')
        ax.set_xlabel('Status')

        img_str = self.plot_to_base64(fig)

        details = pd.DataFrame()
        
        grouped = self.df.groupby('Test_CASE_ID')
        
        agg_data = grouped.agg({
            'duration': 'mean',
            'status': lambda x: (x == 'Passed').mean() * 100
        })

        top_5_fast = agg_data.nsmallest(5, 'duration')
        
        top_5_fast['Remark'] = top_5_fast.apply(
            lambda row: f"Fast execution (avg {row['duration']:.2f}s) with {row['status']:.2f}% pass rate. Potential benchmark for efficiency.", 
            axis=1
        )
        
        top_5_slow = agg_data.nlargest(5, 'duration')
        
        top_5_slow['Remark'] = top_5_slow.apply(
            lambda row: f"Slow execution (avg {row['duration']:.2f}s) with {row['status']:.2f}% pass rate. Investigate for optimization opportunities.", 
            axis=1
        )

        details = pd.concat([top_5_fast, top_5_slow])
        
        details.reset_index(inplace=True)
        
        details.columns = ['Test Case ID', 'Avg Duration (s)', 'Pass Rate (%)', 'Remark']
        
        min_pass_rate = self.config.get('thresholds', 'minimum_pass_rate', default=95.0)

        insights = [
            f"Out of {total_executions} test case executions,... {pass_rate:.2f}% passed with an average duration of {avg_duration:.2f} seconds.",
            f"The longest test took {max_duration:.2f} seconds, while the shortest took {min_duration:.2f} seconds.",
            f"The failure rate is {100-pass_rate:.2f}%, indicating {'room for improvement' if pass_rate < min_pass_rate else 'good performance'} in test reliability.",
            f"There's a {int(max_duration/min_duration) if min_duration > 0 else 'undefined'}x difference between the longest and shortest test durations.",
            f"On average, {total_executions / 365:.0f} test case executions are run daily, suggesting a robust testing pipeline."
        ]

        recommendations = [
            f"Aim to improve the overall pass rate to at least {min_pass_rate}% through targeted test case refinement and bug fixing.",
            f"Investigate and optimize tests taking longer than {avg_duration * 2:.2f} seconds to improve overall efficiency.",
            f"Conduct a detailed analysis of failed tests to identify common failure patterns and root causes.",
            f"Consider parallelizing long-running tests to reduce overall execution time.",
            f"Implement automated alerts for sudden drops in pass rate below {pass_rate-5:.2f}% to catch regressions early."
        ]

        return img_str, insights, recommendations, details

class PassRateTrendAnalysis(AnalysisBase):
    def analyze(self) -> Tuple[str, List[str], List[str], pd.DataFrame]:
        self.df['date'] = self.df['timestamp'].dt.date
        daily_pass_rate = self.df.groupby('date')['status'].apply(lambda x: (x == 'Passed').mean())
        
        X = np.array(range(len(daily_pass_rate))).reshape(-1, 1)
        y = daily_pass_rate.values
        model = LinearRegression().fit(X, y)
        trend = model.coef_[0] * 365  # Annualized trend
        
        fig, ax = plt.subplots(figsize=(
            self.config.get('visualization', 'figure_size', 'width', default=12),
            self.config.get('visualization', 'figure_size', 'height', default=6)
        ))
        
        ax.plot(daily_pass_rate.index, daily_pass_rate.values, label='Daily Pass Rate', color=self.config.get('visualization', 'colors', 'primary', default='blue'))
        ax.plot(daily_pass_rate.index, model.predict(X), color=self.config.get('visualization', 'colors', 'secondary', default='red'), label='Trend')
        
        ax.set_title('Pass Rate Trend Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Pass Rate')
        ax.legend()
        
        img_str = self.plot_to_base64(fig)
        
        min_pass_rate = self.config.get('thresholds', 'minimum_pass_rate', default=95.0)
        
        insights = [
            f"The pass rate is showing an {'improving' if trend > 0 else 'declining'} trend of {trend*100:.2f}% per year.",
            f"The highest daily pass rate was {daily_pass_rate.max()*100:.2f}% on {daily_pass_rate.idxmax()}.",
            f"The lowest daily pass rate was {daily_pass_rate.min()*100:.2f}% on {daily_pass_rate.idxmin()}.",
            f"The average daily pass rate over the period was {daily_pass_rate.mean()*100:.2f}%.",
            f"Pass rate volatility (standard deviation) is {daily_pass_rate.std()*100:.2f}%."
        ]
        
        recommendations = [
            f"{'Continue current practices' if trend > 0 else 'Investigate causes of declining pass rate'} to maintain positive momentum.",
            f"Implement daily pass rate targets of {min(100, daily_pass_rate.mean() + 0.05)*100:.2f}% to drive continuous improvement.",
            f"Conduct root cause analysis for days with pass rates below {daily_pass_rate.quantile(0.1)*100:.2f}% to address systemic issues.",
            "Develop strategies to reduce pass rate volatility and improve consistency.",
            f"Set up automated alerts for any daily pass rate below {(daily_pass_rate.mean() - 2*daily_pass_rate.std())*100:.2f}% to catch significant drops early."
        ]
        
        top_5_pass_rates = daily_pass_rate.nlargest(5).reset_index()
        top_5_pass_rates.columns = ['Date', 'Pass Rate']
        top_5_pass_rates['Pass Rate'] *= 100
        top_5_pass_rates['Remark'] = 'Highest pass rates'
        
        lowest_5_pass_rates = daily_pass_rate.nsmallest(5).reset_index()
        lowest_5_pass_rates.columns = ['Date', 'Pass Rate']
        lowest_5_pass_rates['Pass Rate'] *= 100
        lowest_5_pass_rates['Remark'] = 'Lowest pass rates'
        
        details = pd.concat([top_5_pass_rates, lowest_5_pass_rates])
        
        return img_str, insights, recommendations, details

class AnomalyDetectionAnalysis(AnalysisBase):
    def analyze(self) -> Tuple[str, List[str], List[str], pd.DataFrame]:
        features = ['duration', 'cpu_usage', 'memory_usage', 'response_time']
        X = self.df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        contamination = self.config.get('analysis', 'anomaly_detection', 'contamination', default=0.05)
        clf = IsolationForest(contamination=contamination, random_state=42)
        self.df['anomaly'] = clf.fit_predict(X_scaled)
        
        anomalies = self.df[self.df['anomaly'] == -1]
        
        fig, ax = plt.subplots(figsize=(
            self.config.get('visualization', 'figure_size', 'width', default=10),
            self.config.get('visualization', 'figure_size', 'height', default=6)
        ))
        
        scatter = ax.scatter(self.df['duration'], self.df['cpu_usage'], c=self.df['anomaly'], cmap='viridis')
        ax.set_title('Test Execution Anomalies')
        ax.set_xlabel('Duration (seconds)')
        ax.set_ylabel('CPU Usage (%)')
        plt.colorbar(scatter)
        
        img_str = self.plot_to_base64(fig)
        
        anomaly_rate = len(anomalies) / len(self.df) * 100
        avg_anomaly_duration = anomalies['duration'].mean()
        avg_normal_duration = self.df[self.df['anomaly'] == 1]['duration'].mean()
        
        top_anomalous_tests = anomalies.groupby('Test_CASE_ID').size().sort_values(ascending=False).head()
        top_anomalous_tests = pd.DataFrame({
            'Test Case ID': top_anomalous_tests.index,
            'Anomaly Count': top_anomalous_tests.values,
            'Avg Duration (s)': anomalies.groupby('Test_CASE_ID')['duration'].mean()[top_anomalous_tests.index],
            'Avg CPU Usage (%)': anomalies.groupby('Test_CASE_ID')['cpu_usage'].mean()[top_anomalous_tests.index],
            'Avg Memory Usage (MB)': anomalies.groupby('Test_CASE_ID')['memory_usage'].mean()[top_anomalous_tests.index],
            'Avg Response Time (ms)': anomalies.groupby('Test_CASE_ID')['response_time'].mean()[top_anomalous_tests.index]
        })
        
        top_anomalous_tests['Remark'] = top_anomalous_tests.apply(lambda row: f"High anomaly count ({row['Anomaly Count']}). Unusual resource usage patterns detected. Investigate potential performance issues or environmental factors affecting this test case.", axis=1)
        
        insights = [
            f"Detected {len(anomalies)} potential anomalies ({anomaly_rate:.2f}% of all executions).",
            f"Anomalous tests have an average duration of {avg_anomaly_duration:.2f} seconds, compared to {avg_normal_duration:.2f} seconds for normal tests.",
            f"The most extreme anomaly had a duration of {anomalies['duration'].max():.2f} seconds and CPU usage of {anomalies['cpu_usage'].max():.2f}%.",
            f"{(anomalies['status'] == 'Failed').mean()*100:.2f}% of anomalous tests failed, compared to {(self.df[self.df['anomaly'] == 1]['status'] == 'Failed').mean()*100:.2f}% of normal tests.",
            f"Anomalies are most common in the {anomalies['environment'].mode().values[0]} environment, accounting for {(anomalies['environment'] == anomalies['environment'].mode().values[0]).mean()*100:.2f}% of anomalies."
        ]
        
        recommendations = [
            "Investigate top anomalies, particularly those with extreme duration or resource usage.",
            f"Set up automated alerts for tests exceeding {avg_normal_duration + 3*(self.df[self.df['anomaly'] == 1]['duration'].std()):.2f} seconds duration or {self.df[self.df['anomaly'] == 1]['cpu_usage'].quantile(0.99):.2f}% CPU usage.",
            f"Conduct a detailed review of the {anomalies['environment'].mode().values[0]} environment to understand why it's prone to anomalies.",
            "Implement resource monitoring and logging for all test executions to catch potential issues early.",
            "Develop a machine learning model to predict test execution times and resource usage, flagging significant deviations in real-time."
        ]
        
        return img_str, insights, recommendations, top_anomalous_tests

class TestFlakinessAnalysis(AnalysisBase):
    def analyze(self) -> Tuple[str, List[str], List[str], pd.DataFrame]:
        test_results = self.df.groupby('Test_CASE_ID')['status'].apply(list)
        flaky_tests = test_results[test_results.apply(lambda x: 'Passed' in x and 'Failed' in x)]
        flakiness_scores = flaky_tests.apply(lambda x: min(x.count('Passed'), x.count('Failed')) / len(x))
        
        fig, ax = plt.subplots(figsize=(
            self.config.get('visualization', 'figure_size', 'width', default=10),
            self.config.get('visualization', 'figure_size', 'height', default=6)
        ))
        
        top_flaky_count = self.config.get('analysis', 'test_flakiness', 'top_flaky_tests', default=10)
        flakiness_scores.sort_values(ascending=False).head(top_flaky_count).plot(kind='bar', ax=ax, color=self.config.get('visualization', 'colors', 'primary', default='blue'))
        
        ax.set_title(f'Top {top_flaky_count} Flaky Tests')
        ax.set_ylabel('Flakiness Score')
        ax.set_xlabel('Test Case ID')
        plt.xticks(rotation=45, ha='right')
        
        img_str = self.plot_to_base64(fig)
        
        top_flaky_tests = flakiness_scores.sort_values(ascending=False).head(top_flaky_count)
        top_flaky_tests = pd.DataFrame({
            'Test Case ID': top_flaky_tests.index,
            'Flakiness Score': top_flaky_tests.values,
            'Pass Rate (%)': flaky_tests[top_flaky_tests.index].apply(lambda x: x.count('Passed') / len(x) * 100),
            'Total Executions': flaky_tests[top_flaky_tests.index].apply(len)
        })
        
        top_flaky_tests['Remark'] = top_flaky_tests.apply(lambda row: f"Flakiness score of {row['Flakiness Score']:.4f} indicates inconsistent behavior. " + (f"With {row['Total Executions']} executions and {row['Pass Rate (%)']:.2f}% pass rate, " + f"this test requires immediate attention and potential refactoring.") if row['Flakiness Score'] > self.config.get('thresholds', 'flakiness', 'high', default=0.4) else (f"Moderate flakiness with {row['Total Executions']} executions and {row['Pass Rate (%)']:.2f}% pass rate. " + f"Review test dependencies and environment factors."), axis=1)
        
        insights = [
            f"Detected {len(flaky_tests)} flaky tests out of {len(test_results)} total tests ({len(flaky_tests)/len(test_results)*100:.2f}%).",
            f"The most flaky test ({top_flaky_tests.index[0]}) has a flakiness score of {top_flaky_tests['Flakiness Score'].iloc[0]:.4f}.",
            f"On average, flaky tests have a flakiness score of {flakiness_scores.mean():.4f}.",
            f"The top 10% of flaky tests account for {flakiness_scores.sort_values(ascending=False).head(int(len(flakiness_scores)*0.1)).sum() / flakiness_scores.sum() * 100:.2f}% of total flakiness."
        ]
        
        recommendations = [
            f"Prioritize fixing the top {top_flaky_count} flakiest tests to improve overall test reliability.",
            "Implement retry mechanisms for flaky tests to reduce false negatives.",
            "Analyze common characteristics of flaky tests to prevent introducing new flaky tests.",
            f"Consider quarantining tests with flakiness scores above {self.config.get('thresholds', 'flakiness', 'high', default=0.4)} until they can be fixed or replaced.",
            "Develop guidelines for writing more stable tests based on the characteristics of non-flaky tests."
        ]
        
        return img_str, insights, recommendations, top_flaky_tests

class EnvironmentImpactAnalysis(AnalysisBase):
    def analyze(self) -> Tuple[str, List[str], List[str], pd.DataFrame]:
        env_pass_rates = self.df.groupby('environment')['status'].apply(lambda x: (x == 'Passed').mean() * 100)
        
        fig, ax = plt.subplots(figsize=(
            self.config.get('visualization', 'figure_size', 'width', default=10),
            self.config.get('visualization', 'figure_size', 'height', default=6)
        ))
        
        colors = [self.config.get('visualization', 'colors', 'primary', default='blue'), self.config.get('visualization', 'colors', 'secondary', default='green'), self.config.get('visualization', 'colors', 'tertiary', default='orange')]
        env_pass_rates.plot(kind='bar', ax=ax, color=colors[:len(env_pass_rates)])
        
        ax.set_title('Pass Rates by Environment')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_xlabel('Environment')
        plt.xticks(rotation=45, ha='right')
        
        img_str = self.plot_to_base64(fig)
        
        env_details = self.df.groupby('environment').agg({
            'status': lambda x: (x == 'Passed').mean() * 100,
            'duration': 'mean',
            'Test_CASE_ID': 'count'
        }).sort_values('status', ascending=False)
        
        env_details = env_details.reset_index()
        env_details.columns = ['Environment', 'Pass Rate (%)', 'Avg Duration (s)', 'Total Executions']
        env_details['Remark'] = env_details.apply(lambda row: f"{'High' if row['Pass Rate (%)'] > env_details['Pass Rate (%)'].mean() else 'Low'} pass rate environment", axis=1)
        
        insights = [
            f"The environment with the highest pass rate is {env_pass_rates.idxmax()} at {env_pass_rates.max():.2f}%.",
            f"The environment with the lowest pass rate is {env_pass_rates.idxmin()} at {env_pass_rates.min():.2f}%.",
            f"There's a {env_pass_rates.max() - env_pass_rates.min():.2f}% difference between the best and worst performing environments.",
            f"The average pass rate across all environments is {env_pass_rates.mean():.2f}%.",
            f"{len(env_pass_rates[env_pass_rates > env_pass_rates.mean()])} out of {len(env_pass_rates)} environments perform above average."
        ]

        recommendations = [
            f"Investigate and improve the {env_pass_rates.idxmin()} environment to match the performance of {env_pass_rates.idxmax()}.",
            "Standardize the configuration of the top-performing environments across all test environments.",
            f"Conduct a detailed analysis of tests that fail in {env_pass_rates.idxmin()} but pass in {env_pass_rates.idxmax()} to identify environment-specific issues.",
            f"Implement continuous monitoring for environment health, with alerts for pass rates dropping below {env_pass_rates.mean():.2f}%.",
            "Develop a playbook for quickly diagnosing and addressing environment-specific test failures."
        ]

        return img_str, insights, recommendations, env_details

class CodeChangesImpactAnalysis(AnalysisBase):
    def analyze(self) -> Tuple[str, List[str], List[str], pd.DataFrame, str]:
        total_lines_changed = self.df['lines_added'] + self.df['lines_removed']
        
        correlation = total_lines_changed.corr(self.df['status'].map({'Passed': 1, 'Failed': 0}))

        large_change_threshold = self.config.get('thresholds', 'large_change_lines', default=500)
        bins = [0, 10, 50, 100, large_change_threshold, float('inf')]
        labels = ['1-10', '11-50', '51-100', f'101-{large_change_threshold}', f'{large_change_threshold}+']
        self.df['change_category'] = pd.cut(total_lines_changed, bins=bins, labels=labels, include_lowest=True)

        pass_rates = self.df.groupby('change_category')['status'].apply(lambda x: (x == 'Passed').mean() * 100)

        fig, ax = plt.subplots(figsize=(
            self.config.get('visualization', 'figure_size', 'width', default=12),
            self.config.get('visualization', 'figure_size', 'height', default=6)
        ))
        bars = ax.bar(pass_rates.index, pass_rates.values, color=self.config.get('visualization', 'colors', 'primary', default='skyblue'))
        ax.set_title('Pass Rate by Number of Lines Changed', fontsize=16)
        ax.set_xlabel('Number of Lines Changed', fontsize=12)
        ax.set_ylabel('Pass Rate (%)', fontsize=12)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

        plt.xticks(rotation=0)
        img_str = self.plot_to_base64(fig)

        change_impact = self.df.groupby('change_category').agg({
            'status': lambda x: (x == 'Passed').mean() * 100,
            'lines_added': 'mean',
            'lines_removed': 'mean',
            'Test_CASE_ID': 'count'
        }).reset_index()
        
        change_impact.columns = ['Lines Changed', 'Pass Rate (%)', 'Avg Lines Added', 'Avg Lines Removed', 'Total Executions']
        change_impact['Remark'] = change_impact.apply(lambda row: 
            f"{'High' if row['Pass Rate (%)'] > change_impact['Pass Rate (%)'].mean() else 'Low'} pass rate for this change range. " +
            f"{'Consider breaking changes into smaller commits' if row['Avg Lines Added'] + row['Avg Lines Removed'] > total_lines_changed.mean() else 'Maintain this change size for stability'}.", axis=1)

        insights = [
            f"Correlation between lines changed and test success: {correlation:.2f}",
            f"Changes with 1-10 lines have a {pass_rates['1-10']:.1f}% pass rate.",
            f"Large changes ({large_change_threshold}+ lines) have a {pass_rates[f'{large_change_threshold}+']:.1f}% pass rate.",
            f"The category with the highest pass rate is {pass_rates.idxmax()} lines changed, at {pass_rates.max():.1f}%.",
            f"The category with the lowest pass rate is {pass_rates.idxmin()} lines changed, at {pass_rates.min():.1f}%."
        ]

        recommendations = [
            f"{'Consider breaking down large changes' if pass_rates[f'{large_change_threshold}+'] < pass_rates['1-10'] else 'Maintain current practices'} to minimize the impact on test success.",
            f"Implement a code review policy for changes exceeding {labels[2]} lines to ensure quality.",
            f"Set up automated alerts for commits with more than {labels[3]} lines changed, as they may need extra testing.",
            "Encourage smaller, more frequent commits to reduce the risk of test failures.",
            f"Investigate why changes in the {pass_rates.idxmin()} category have a lower pass rate and adjust development practices accordingly."
        ]
        table_explanation = ("How to read this table: 'Lines Changed' shows the range of code lines modified. " 
                             "'Pass Rate (%)' is the percentage of tests that passed for each range. "
                             "'Avg Lines Added/Removed' show the average number of lines added or removed. "
                             "'Total Executions' is the number of tests run for each range. "
                             "The 'Remark' column provides a brief analysis of each category.")

        return img_str, insights, recommendations, change_impact, table_explanation

class FuturePerformancePredictionAnalysis(AnalysisBase):
    def analyze(self) -> Tuple[str, List[str], List[str], pd.DataFrame]:
        self.df['date'] = pd.to_datetime(self.df['timestamp']).dt.date
        daily_pass_rate = self.df.groupby('date')['status'].apply(lambda x: (x == 'Passed').mean())
        
        X = np.array(range(len(daily_pass_rate))).reshape(-1, 1)
        y = daily_pass_rate.values
        model = LinearRegression().fit(X, y)
        
        last_date = daily_pass_rate.index[-1]
        prediction_days = self.config.get('analysis', 'future_prediction', 'days', default=30)
        next_n_days = pd.date_range(last_date + timedelta(days=1), periods=prediction_days)
        X_future = np.array(range(len(daily_pass_rate), len(daily_pass_rate) + prediction_days)).reshape(-1, 1)
        future_pass_rates = model.predict(X_future)
        
        n = len(X)
        dof = n - 2
        t_value = t.ppf(0.975, dof)
        mse = np.sum((y - model.predict(X)) ** 2) / dof
        std_errors = np.sqrt(mse * (1 + 1/n + (X_future - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2)))
        ci = t_value * std_errors.flatten()
        
        fig, ax = plt.subplots(figsize=(
            self.config.get('visualization', 'figure_size', 'width', default=12),
            self.config.get('visualization', 'figure_size', 'height', default=6)
        ))
        ax.plot(daily_pass_rate.index, daily_pass_rate.values, label='Historical Pass Rate', 
                color=self.config.get('visualization', 'colors', 'primary', default='blue'))
        ax.plot(next_n_days, future_pass_rates, color=self.config.get('visualization', 'colors', 'secondary', default='green'), 
                linestyle='--', label='Predicted Pass Rate')
        ax.fill_between(next_n_days, future_pass_rates - ci, future_pass_rates + ci, 
                        color=self.config.get('visualization', 'colors', 'tertiary', default='green'), 
                        alpha=0.2, label='95% Confidence Interval')
        ax.set_title(f'Pass Rate Prediction for Next {prediction_days} Days')
        ax.set_xlabel('Date')
        ax.set_ylabel('Pass Rate')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        img_str = self.plot_to_base64(fig)

        prediction_data = pd.DataFrame({
            'Date': next_n_days,
            'Predicted Pass Rate (%)': future_pass_rates * 100,
            'Lower CI (%)': (future_pass_rates - ci) * 100,
            'Upper CI (%)': (future_pass_rates + ci) * 100
        })
        prediction_data = prediction_data.iloc[::3]  # Take every 3rd row
        prediction_data['Remark'] = prediction_data.apply(lambda row: 
            f"Expected pass rate between {row['Lower CI (%)']:.2f}% and {row['Upper CI (%)']:.2f}%. " +
            ("Consider allocating more resources if trend continues." if row['Predicted Pass Rate (%)'] < 90 else 
             "Maintain current practices to sustain high performance."), axis=1)

        insights = [
            f"Predicted pass rate in {prediction_days} days: {future_pass_rates[-1]*100:.2f}%",
            f"Best case scenario (upper CI): {(future_pass_rates + ci)[-1]*100:.2f}% pass rate",
            f"Worst case scenario (lower CI): {(future_pass_rates - ci)[-1]*100:.2f}% pass rate",
            f"The model predicts a {'positive' if future_pass_rates[-1] > y[-1] else 'negative'} trend in pass rates over the next month.",
            f"Prediction uncertainty (CI width) starts at ±{(ci[0]*2*100):.2f}% and ends at ±{(ci[-1]*2*100):.2f}%"
        ]

        recommendations = [
            f"Set a goal to achieve at least {min(100, future_pass_rates[-1]*100 + 2):.2f}% pass rate within the next {prediction_days} days.",
            "Implement daily monitoring of actual pass rates against predicted values to catch deviations early.",
            f"Prepare contingency plans for potential pass rate drops below {(future_pass_rates - ci).min()*100:.2f}%.",
            "Conduct a detailed review of test infrastructure and processes to sustain the predicted improvement trend.",
            "Allocate additional resources to testing if the actual pass rate falls below the predicted rate for 3 consecutive days."
        ]

        return img_str, insights, recommendations, prediction_data

class ReportGenerator:
    def __init__(self, df: pd.DataFrame, config: ConfigManager):
        self.df = df
        self.config = config

    def generate(self) -> List[Dict[str, Any]]:
        analyses = [
            ("Overall Insights", OverallInsightsAnalysis(self.df, self.config).analyze),
            ("Pass Rate Trend", PassRateTrendAnalysis(self.df, self.config).analyze),
            ("Anomaly Detection", AnomalyDetectionAnalysis(self.df, self.config).analyze),
            ("Test Flakiness", TestFlakinessAnalysis(self.df, self.config).analyze),
            ("Environment Impact", EnvironmentImpactAnalysis(self.df, self.config).analyze),
            ("Code Changes Impact", CodeChangesImpactAnalysis(self.df, self.config).analyze),
            ("Future Performance Prediction", FuturePerformancePredictionAnalysis(self.df, self.config).analyze)
        ]

        report_data = []
        for title, analysis_func in analyses:
            try:
                logger.info(f"Starting {title} analysis...")
                result = analysis_func()
                if len(result) == 4:
                    img_str, insights, recommendations, details = result
                    table_explanation = None
                else:
                    img_str, insights, recommendations, details, table_explanation = result

                report_data.append({
                    "title": title,
                    "img_str": img_str,
                    "insights": insights,
                    "recommendations": recommendations,
                    "details": details.to_html(index=False) if isinstance(details, pd.DataFrame) else None,
                    "table_explanation": table_explanation
                })
                logger.info(f"Completed {title} analysis")
            except Exception as e:
                logger.error(f"Error in {title} analysis: {str(e)}", exc_info=True)

        return report_data

    def generate_html_report(self, template_path: str, output_file: str):
        report_data = self.generate()
        with open(template_path, 'r') as f:
            template = Template(f.read())
        
        html_content = template.render(
            report_data=report_data,
            company_name=self.config.get('report', 'company_name', default='Your Company'),
            report_title=self.config.get('report', 'title', default='AI-Driven Test Insights & Recommendations')
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

def main():
    try:
        config = ConfigManager('config.yaml')
        logger.info("Starting AI-driven test insights analysis...")

        df = DataProcessor.load_data(config.get('data_file'))
        if not DataValidator.validate_csv(df):
            raise ValueError("Invalid input data")

        df = DataProcessor.preprocess_data(df)

        report_generator = ReportGenerator(df, config)
        report_generator.generate_html_report(
            config.get('report_template'),
            config.get('output_file')
        )

        logger.info(f"AI-driven test insights report generated as {config.get('output_file')}")
    except Exception as e:
        logger.error(f"An error occurred during the AI-driven test insights analysis: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()                                                      
