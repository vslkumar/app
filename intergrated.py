import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import sys
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend to path
sys.path.append('../backend')
sys.path.append('backend')

from backend.api import FastAPIBackend
from backend.data_generators import generate_sample_logs, generate_sample_metrics, generate_sample_alerts
from utils.helpers import format_timestamp, get_timeframe_options, get_severity_icon

st.set_page_config(
    page_title="Dashboard Stories - AI Observability",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dashboard Storytelling Engine")
st.markdown("AI-generated narratives that tell the story of your system's behavior and performance.")

# Error patterns to inject 4 times a day
ERROR_PATTERNS = [
    {"type": "Request Timeout", "codes": ["504", "ETIMEDOUT"], "impact": ["response_time", "error_rate"]},
    {"type": "SSL/TLS Errors", "codes": ["ERR_CERT_AUTHORITY_INVALID", "SSL_ERROR"], "impact": ["error_rate"]},
    {"type": "Connection Refused", "codes": ["ECONNREFUSED"], "impact": ["error_rate"]},
    {"type": "4xx Errors", "codes": ["400", "401", "403", "404", "405", "409", "413", "429"], "impact": ["error_rate"]},
    {"type": "5xx Errors", "codes": ["500", "502", "503", "504"], "impact": ["response_time", "error_rate"]}
]

def generate_time_series_data(timeframe):
    """Generate synthetic time series data with error patterns"""
    end_time = datetime.now()
    
    # Set start time based on timeframe
    if timeframe == "15m":
        start_time = end_time - timedelta(minutes=15)
    elif timeframe == "1h":
        start_time = end_time - timedelta(hours=1)
    elif timeframe == "24h":
        start_time = end_time - timedelta(hours=24)
    else:
        start_time = end_time - timedelta(hours=1)  # default
    
    timestamps = pd.date_range(start_time, end_time, freq='5min')
    
    # Base values
    cpu_base = np.random.normal(30, 5, len(timestamps))
    memory_usage = np.clip(np.random.normal(45, 8, len(timestamps)), 0, 100)
    response_time = np.clip(np.random.gamma(2, 10, len(timestamps)), 5, 500)
    error_rate = np.clip(np.random.exponential(0.5, len(timestamps)), 0, 10)
    
    # Create error windows (4 times in the timeframe, each lasting ~15 minutes)
    error_windows = []
    for _ in range(4):
        window_start = random.choice(timestamps)
        window_end = window_start + timedelta(minutes=15)
        error_windows.append((window_start, window_end))
    
    # Apply error patterns during error windows
    for ts_idx, ts in enumerate(timestamps):
        in_error_window = any(start <= ts <= end for start, end in error_windows)
        
        if in_error_window:
            error_pattern = random.choice(ERROR_PATTERNS)
            
            if "response_time" in error_pattern["impact"]:
                response_time[ts_idx] = np.clip(response_time[ts_idx] * random.uniform(3, 10), 100, 2000)
            
            if "error_rate" in error_pattern["impact"]:
                error_rate[ts_idx] = np.clip(error_rate[ts_idx] * random.uniform(5, 20), 5, 100)
            
            # CPU spikes during errors
            cpu_base[ts_idx] = np.clip(cpu_base[ts_idx] * random.uniform(1.5, 3), 0, 100)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_base,
        'memory_usage': memory_usage,
        'response_time': response_time,
        'error_rate': error_rate,
        'service': ['frontend-app' if i % 2 == 0 else 'backend-service' for i in range(len(timestamps))]
    })
    
    return df, error_windows

# Initialize backend
if 'api_backend' not in st.session_state:
    st.session_state.api_backend = FastAPIBackend()

# Sidebar controls
st.sidebar.header("Story Configuration")

# Timeframe selection
timeframe_options = get_timeframe_options()
timeframe = st.sidebar.selectbox(
    "Time Period",
    options=[opt["value"] for opt in timeframe_options],
    format_func=lambda x: next(opt["label"] for opt in timeframe_options if opt["value"] == x),
    index=1  # Default to "Last 1 hour"
)

# Story type selection
story_type = st.sidebar.selectbox(
    "Story Focus",
    ["Overall Health", "Error Analysis", "Performance", "Service Activity", "Custom"],
    help="Choose what aspect of the system to focus the story on"
)

# Data sources to include
st.sidebar.subheader("Data Sources")
include_logs = st.sidebar.checkbox("Include Log Data", value=True)
include_metrics = st.sidebar.checkbox("Include Performance Metrics", value=True)
include_alerts = st.sidebar.checkbox("Include Alert Data", value=True)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5 minutes)", value=False)

if auto_refresh:
    st.sidebar.info("🔄 Auto-refresh enabled. The story will update automatically.")

# Generate story button
if st.button("📚 Generate Story", type="primary") or auto_refresh:
    
    with st.spinner("🧠 Analyzing system data and crafting your story..."):
        try:
            # Collect data from different sources
            story_data = {}
            
            if include_logs:
                # Generate log data for the timeframe
                log_count = 500 if timeframe in ["1h", "15m"] else 1000
                logs = generate_sample_logs(log_count)
                story_data['logs'] = logs
            
            if include_metrics:
                # Generate metrics data with error patterns
                metrics, error_windows = generate_time_series_data(timeframe)
                story_data['metrics'] = metrics
                story_data['error_windows'] = error_windows
            
            if include_alerts:
                # Generate alert data
                alerts = generate_sample_alerts()
                story_data['alerts'] = alerts
            
            # Generate AI story
            async def generate_story():
                return await st.session_state.api_backend.generate_story_summary(timeframe)
            
            story_result = asyncio.run(generate_story())
            
            # Store results in session state
            st.session_state.story_result = story_result
            st.session_state.story_data = story_data
            st.session_state.story_timeframe = timeframe
            
        except Exception as e:
            st.error(f"Error generating story: {str(e)}")

# Display story and dashboard
if 'story_result' in st.session_state and 'story_data' in st.session_state:
    story = st.session_state.story_result
    data = st.session_state.story_data
    
    if story['status'] == 'success':
        # Main story section
        st.subheader("📖 System Story")
        
        # Story card with nice formatting
        story_container = st.container()
        with story_container:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h3 style="margin-top: 0; color: white;">🎭 {st.session_state.story_timeframe.upper()} System Narrative</h3>
                <div style="font-size: 1.1em; line-height: 1.6; font-style: italic;">
                    "{story['story']}"
                </div>
                <div style="margin-top: 1rem; font-size: 0.9em; opacity: 0.8;">
                    📊 Based on {story.get('log_count', 0)} logs and system patterns
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Key metrics overview
        st.subheader("📊 Key Metrics Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if 'logs' in data:
            logs_df = pd.DataFrame(data['logs'])
            
            with col1:
                total_logs = len(logs_df)
                st.metric("Total Events", f"{total_logs:,}")
            
            with col2:
                error_logs = len(logs_df[logs_df['level'] == 'ERROR'])
                error_rate = (error_logs / total_logs * 100) if total_logs > 0 else 0
                st.metric("Error Rate", f"{error_rate:.1f}%", delta=f"{error_logs} errors")
            
            with col3:
                unique_services = logs_df['service'].nunique()
                st.metric("Active Services", unique_services)
            
            with col4:
                if 'alerts' in data:
                    active_alerts = len([a for a in data['alerts'] if a['status'] == 'ACTIVE'])
                    st.metric("Active Alerts", active_alerts)
                else:
                    st.metric("Active Alerts", "0")
        
        # Detailed visualizations
        st.subheader("📈 Detailed Analysis")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Activity Timeline", "⚠️ Alerts & Issues", "📊 Performance", "🔍 Deep Dive"])
        
        with tab1:
            # Activity timeline
            if 'logs' in data:
                st.markdown("**Log Activity Over Time**")
                
                logs_df = pd.DataFrame(data['logs'])
                logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
                logs_df['time_bucket'] = logs_df['timestamp'].dt.floor('10T')  # 10-minute buckets
                
                # Count by level and time
                timeline_data = logs_df.groupby(['time_bucket', 'level']).size().reset_index(name='count')
                
                fig = px.bar(
                    timeline_data,
                    x='time_bucket',
                    y='count',
                    color='level',
                    title="Log Activity Timeline (10-minute intervals)",
                    color_discrete_map={
                        'ERROR': '#ff4444',
                        'WARN': '#ffaa00',
                        'INFO': '#44aa44',
                        'DEBUG': '#4444ff'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Service activity heatmap
                st.markdown("**Service Activity Heatmap**")
                
                service_time_data = logs_df.groupby([
                    logs_df['timestamp'].dt.floor('H'), 
                    'service'
                ]).size().reset_index(name='count')
                
                if not service_time_data.empty:
                    pivot_data = service_time_data.pivot(index='service', columns='timestamp', values='count').fillna(0)
                    
                    fig_heatmap = px.imshow(
                        pivot_data.values,
                        x=[t.strftime('%H:%M') for t in pivot_data.columns],
                        y=pivot_data.index,
                        title="Service Activity Heatmap (Hourly)",
                        color_continuous_scale='Viridis'
                    )
                    fig_heatmap.update_layout(height=400)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab2:
            # Alerts and issues
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'alerts' in data:
                    st.markdown("**Recent Alerts**")
                    
                    alerts_df = pd.DataFrame(data['alerts'])
                    
                    for alert in data['alerts'][:5]:  # Show top 5 alerts
                        severity_color = {
                            'CRITICAL': '#ff0000',
                            'HIGH': '#ff4444', 
                            'MEDIUM': '#ffaa00',
                            'LOW': '#44aa44'
                        }.get(alert['severity'], '#666666')
                        
                        status_emoji = {
                            'ACTIVE': '🔴',
                            'RESOLVED': '✅',
                            'ACKNOWLEDGED': '🟡'
                        }.get(alert['status'], '⚪')
                        
                        st.markdown(f"""
                        <div style="border-left: 4px solid {severity_color}; padding: 10px; margin: 10px 0; background-color: #f8f9fa;">
                            <strong>{status_emoji} {alert['type']}</strong> - {alert['service']}<br>
                            <small>{format_timestamp(alert['timestamp'], 'relative')}</small><br>
                            {alert['message']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No alert data available")
            
            with col2:
                # Alert summary
                if 'alerts' in data and data['alerts']:
                    st.markdown("**Alert Summary**")
                    
                    alerts_df = pd.DataFrame(data['alerts'])
                    
                    # By severity
                    severity_counts = alerts_df['severity'].value_counts()
                    for severity, count in severity_counts.items():
                        icon = get_severity_icon(severity)
                        st.metric(f"{icon} {severity}", count)
                    
                    # By status
                    st.markdown("**By Status**")
                    status_counts = alerts_df['status'].value_counts()
                    for status, count in status_counts.items():
                        emoji = {'ACTIVE': '🔴', 'RESOLVED': '✅', 'ACKNOWLEDGED': '🟡'}.get(status, '⚪')
                        st.write(f"{emoji} {status}: {count}")
        
        with tab3:
            # Performance metrics
            if 'metrics' in data:
                st.markdown("**System Performance Metrics**")
                
                metrics_df = pd.DataFrame(data['metrics'])
                error_windows = data.get('error_windows', [])
                
                # Metrics overview
                st.subheader("Key Performance Indicators")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg CPU Usage", f"{metrics_df['cpu_usage'].mean():.1f}%")
                col2.metric("Avg Memory Usage", f"{metrics_df['memory_usage'].mean():.1f}%")
                col3.metric("Avg Response Time", f"{metrics_df['response_time'].mean():.1f} ms")
                col4.metric("Avg Error Rate", f"{metrics_df['error_rate'].mean():.2f}%")
                
                # Time series charts
                st.subheader("Performance Over Time")
                perf_tab1, perf_tab2, perf_tab3, perf_tab4 = st.tabs(["CPU Usage", "Memory Usage", "Response Time", "Error Rate"])
                
                with perf_tab1:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(data=metrics_df, x='timestamp', y='cpu_usage', hue='service', ax=ax)
                    
                    # Highlight error windows
                    for start, end in error_windows:
                        ax.axvspan(start, end, color='red', alpha=0.1)
                    
                    ax.set_title("CPU Usage Over Time (Red areas = error periods)")
                    ax.set_ylabel("CPU Usage (%)")
                    st.pyplot(fig)
                
                with perf_tab2:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(data=metrics_df, x='timestamp', y='memory_usage', hue='service', ax=ax)
                    
                    # Highlight error windows
                    for start, end in error_windows:
                        ax.axvspan(start, end, color='red', alpha=0.1)
                    
                    ax.set_title("Memory Usage Over Time (Red areas = error periods)")
                    ax.set_ylabel("Memory Usage (%)")
                    st.pyplot(fig)
                
                with perf_tab3:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(data=metrics_df, x='timestamp', y='response_time', hue='service', ax=ax)
                    
                    # Highlight error windows
                    for start, end in error_windows:
                        ax.axvspan(start, end, color='red', alpha=0.1)
                    
                    ax.set_title("Response Time Over Time (Red areas = error periods)")
                    ax.set_ylabel("Response Time (ms)")
                    st.pyplot(fig)
                
                with perf_tab4:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(data=metrics_df, x='timestamp', y='error_rate', hue='service', ax=ax)
                    
                    # Highlight error windows
                    for start, end in error_windows:
                        ax.axvspan(start, end, color='red', alpha=0.1)
                    
                    ax.set_title("Error Rate Over Time (Red areas = error periods)")
                    ax.set_ylabel("Error Rate (%)")
                    st.pyplot(fig)
            else:
                st.info("No performance metrics available")
        
        with tab4:
            # Deep dive analysis
            st.markdown("**Deep Dive Analysis**")
            
            if 'patterns' in story:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Log Patterns Detected**")
                    patterns = story['patterns']
                    
                    if 'top_errors' in patterns:
                        st.markdown("**Most Common Errors:**")
                        for error, count in list(patterns['top_errors'].items())[:5]:
                            st.write(f"• {error}: {count} occurrences")
                    
                    if 'services' in patterns:
                        st.markdown("**Most Active Services:**")
                        for service, count in list(patterns['services'].items())[:5]:
                            st.write(f"• {service}: {count} logs")
                
                with col2:
                    st.markdown("**System Health Indicators**")
                    
                    # Health score calculation
                    error_rate = patterns.get('error_rate', 0)
                    if error_rate < 1:
                        health_score = "🟢 Excellent"
                    elif error_rate < 5:
                        health_score = "🟡 Good"
                    elif error_rate < 10:
                        health_score = "🟠 Fair"
                    else:
                        health_score = "🔴 Poor"
                    
                    st.write(f"**Overall Health:** {health_score}")
                    st.write(f"**Error Rate:** {error_rate:.1f}%")
                    st.write(f"**Total Events:** {patterns.get('total_logs', 0):,}")
                    st.write(f"**Active Services:** {len(patterns.get('services', {}))}")
            
            # Raw data access
            st.markdown("**📊 Raw Data Export**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Export Story Report"):
                    story_report = {
                        "timestamp": datetime.now().isoformat(),
                        "timeframe": st.session_state.story_timeframe,
                        "story": story['story'],
                        "patterns": story.get('patterns', {}),
                        "metrics_summary": "included" if 'metrics' in data else "not_included"
                    }
                    
                    st.download_button(
                        "💾 Download Report",
                        data=pd.Series([story_report]).to_json(indent=2),
                        file_name=f"story_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if 'logs' in data and st.button("📋 Export Log Data"):
                    logs_csv = pd.DataFrame(data['logs']).to_csv(index=False)
                    st.download_button(
                        "💾 Download Logs CSV",
                        data=logs_csv,
                        file_name=f"story_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("🔄 Generate New Story"):
                    # Clear session state to force regeneration
                    if 'story_result' in st.session_state:
                        del st.session_state.story_result
                    if 'story_data' in st.session_state:
                        del st.session_state.story_data
                    st.rerun()
    
    else:
        st.error(f"Failed to generate story: {story['message']}")

else:
    # Initial state - show what the storytelling engine does
    st.info("👆 Configure your story parameters and click 'Generate Story' to begin.")
    
    # Feature explanation
    st.subheader("🎭 What is Dashboard Storytelling?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 AI-Powered Narratives**
        - Converts complex data into human-readable stories
        - Identifies trends, anomalies, and patterns
        - Provides context and insights automatically
        
        **📊 Multi-Source Analysis**
        - Combines logs, metrics, and alerts
        - Cross-correlates events across services
        - Highlights significant system events
        """)
    
    with col2:
        st.markdown("""
        **⚡ Real-Time Intelligence**
        - Updates stories as new data arrives
        - Adapts narrative based on system state
        - Prioritizes most important findings
        
        **🎯 Actionable Insights**
        - Explains what happened and why
        - Suggests areas needing attention
        - Helps prioritize investigation efforts
        """)
    
    # Sample story preview
    st.subheader("📖 Sample Story")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        font-style: italic;
    ">
        "During the last hour, the system processed 2,847 events with a healthy 1.2% error rate. 
        The payment-api service experienced a brief spike in response times around 14:30, 
        triggering 3 retry attempts that were successfully resolved. All critical services 
        maintained normal operation levels with no active alerts."
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📚 AI Storytelling powered by GPT-4 | Turn data into insights with natural language"
    "</div>", 
    unsafe_allow_html=True
)
