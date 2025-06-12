import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime, timedelta
import json
import random

# Set page config
st.set_page_config(
    page_title="Dynatrace Data Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def save_logs_to_file(logs, filename="logs.txt"):
    """Save logs to a text file in JSON format"""
    with open(filename, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')
    st.sidebar.success(f"Logs saved to {filename}")

# Error patterns to inject 4 times a day
ERROR_PATTERNS = [
    {"type": "Request Timeout", "codes": ["504", "ETIMEDOUT"], "impact": ["response_time", "error_rate"]},
    {"type": "SSL/TLS Errors", "codes": ["ERR_CERT_AUTHORITY_INVALID", "SSL_ERROR"], "impact": ["error_rate"]},
    {"type": "Connection Refused", "codes": ["ECONNREFUSED"], "impact": ["error_rate"]},
    {"type": "4xx Errors", "codes": ["400", "401", "403", "404", "405", "409", "413", "429"], "impact": ["error_rate"]},
    {"type": "5xx Errors", "codes": ["500", "502", "503", "504"], "impact": ["response_time", "error_rate"]}
]

# Generate synthetic data functions with error injection
def generate_time_series_data():
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    timestamps = pd.date_range(start_time, end_time, freq='5min')
    
    # Base values
    cpu_base = np.random.normal(30, 5, len(timestamps))
    memory_usage = np.clip(np.random.normal(45, 8, len(timestamps)), 0, 100)
    response_time = np.clip(np.random.gamma(2, 10, len(timestamps)), 5, 500)
    error_rate = np.clip(np.random.exponential(0.5, len(timestamps)), 0, 10)
    
    # Create error windows (4 times a day, each lasting ~15 minutes)
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
        'service': ['operation-user-interface-app' if i % 2 == 0 else 'document-notification-api' for i in range(len(timestamps))]
    })
    
    return df, error_windows

def generate_splunk_logs(error_windows):
    logs = []
    services = ['operation-user-interface-app', 'document-notification-api', 'payment', 'auth', 'database']
    log_levels = ['INFO', 'WARN', 'ERROR', 'DEBUG']
    http_methods = ['GET', 'POST', 'PUT', 'DELETE']
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Mozilla/5.0 (Linux; Android 10; SM-G981B)'
    ]
    
    # Generate normal logs
    for _ in range(500):
        timestamp = datetime.now() - timedelta(minutes=random.randint(1, 1440))
        service = random.choice(services)
        log_level = random.choices(log_levels, weights=[0.6, 0.2, 0.15, 0.05])[0]
        
        log = {
            "timestamp": timestamp.isoformat(),
            "service": service,
            "level": log_level,
            "message": f"{random.choice(['Processing request', 'Completed request', 'Starting transaction', 'Finished operation'])} for {random.choice(['/api/users', '/api/orders', '/auth/login', '/products'])}",
            "method": random.choice(http_methods),
            "status_code": random.choices([200, 201, 400, 401, 403, 404, 500, 502, 503], 
                                         weights=[0.4, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05])[0],
            "response_time_ms": random.randint(10, 500),
            "user_agent": random.choice(user_agents),
            "source_ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        }
        logs.append(log)
    
    # Generate error logs during error windows
    for window_start, window_end in error_windows:
        window_duration = (window_end - window_start).total_seconds()
        error_logs_count = random.randint(10, 30)
        
        for _ in range(error_logs_count):
            seconds_in_window = random.uniform(0, window_duration)
            timestamp = window_start + timedelta(seconds=seconds_in_window)
            service = random.choice(services)
            error_pattern = random.choice(ERROR_PATTERNS)
            
            if error_pattern["type"] in ["Request Timeout", "Connection Refused"]:
                message = f"{error_pattern['type']} occurred while connecting to {service} service"
                status_code = "000"
            elif error_pattern["type"] == "SSL/TLS Errors":
                message = f"SSL Handshake failed: {random.choice(error_pattern['codes'])}"
                status_code = "000"
            else:
                status_code = random.choice(error_pattern["codes"])
                message = f"HTTP {status_code} error for {random.choice(['/api/users', '/api/orders', '/auth/login', '/products'])}"
            
            log = {
                "timestamp": timestamp.isoformat(),
                "service": service,
                "level": "ERROR",
                "message": message,
                "method": random.choice(http_methods),
                "status_code": status_code,
                "response_time_ms": random.randint(1000, 10000) if error_pattern["type"] == "Request Timeout" else 0,
                "user_agent": random.choice(user_agents),
                "source_ip": f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "error_type": error_pattern["type"],
                "stack_trace": "..." if random.random() > 0.7 else None
            }
            logs.append(log)
    
    # Sort logs by timestamp
    logs.sort(key=lambda x: x["timestamp"])
    return logs

# ... [keep all the other existing functions like generate_topology_data, generate_problems_data, generate_user_sessions the same] ...
def generate_topology_data():
    services = [
        {'id': 'operation-user-interface-app', 'name': 'Web Frontend', 'type': 'APPLICATION', 'health': 'HEALTHY'},
        {'id': 'document-notification-api', 'name': 'Backend Service', 'type': 'SERVICE', 'health': 'HEALTHY'},
        {'id': 'database', 'name': 'Customer DB', 'type': 'DATABASE', 'health': 'HEALTHY'},
        {'id': 'payment', 'name': 'Payment Service', 'type': 'SERVICE', 'health': 'UNHEALTHY'},
        {'id': 'auth', 'name': 'Auth Service', 'type': 'SERVICE', 'health': 'HEALTHY'}
    ]
    
    relationships = [
        {'from': 'operation-user-interface-app', 'to': 'document-notification-api', 'request_count': 1200, 'error_rate': 0.5},
        {'from': 'document-notification-api', 'to': 'database', 'request_count': 2400, 'error_rate': 0.2},
        {'from': 'document-notification-api', 'to': 'payment', 'request_count': 800, 'error_rate': 5.2},
        {'from': 'document-notification-api', 'to': 'auth', 'request_count': 1500, 'error_rate': 1.1},
        {'from': 'operation-user-interface-app', 'to': 'auth', 'request_count': 300, 'error_rate': 0.3}
    ]
    
    return {
        'nodes': services,
        'edges': relationships
    }
def generate_problems_data():
    problems = []
    severities = ['AVAILABILITY', 'ERROR', 'PERFORMANCE', 'RESOURCE']
    
    for i in range(10):
        start_time = datetime.now() - timedelta(hours=np.random.randint(1, 72))
        end_time = start_time + timedelta(minutes=np.random.randint(5, 240)) if np.random.random() > 0.3 else None
        
        problems.append({
            'problem_id': f'P-{10000 + i}',
            'title': f'High response time in {np.random.choice(["operation-user-interface-app", "document-notification-api", "payment"])}',
            'severity': np.random.choice(severities),
            'status': 'OPEN' if end_time is None else 'CLOSED',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat() if end_time else None,
            'affected_services': np.random.choice(['operation-user-interface-app', 'document-notification-api', 'database', 'payment', 'auth'], 
                                               size=np.random.randint(1, 4), replace=False).tolist(),
            'impact': np.random.choice(['SERVICE', 'APPLICATION', 'ENVIRONMENT']),
            'root_cause': np.random.choice([
                'Memory leak detected',
                'Database connection pool exhausted',
                'High CPU utilization',
                'Network latency between zones',
                'Third-party API degradation'
            ])
        })
    
    return problems

def generate_user_sessions():
    sessions = []
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36'
    ]
    
    countries = ['US', 'GB', 'DE', 'FR', 'JP', 'IN', 'BR', 'CA', 'AU']
    
    for i in range(50):
        start_time = datetime.now() - timedelta(minutes=np.random.randint(1, 1440))
        duration = np.random.randint(5, 600)
        has_error = np.random.random() > 0.8
        
        sessions.append({
            'session_id': f'session-{1000 + i}',
            'user_id': f'user-{np.random.randint(100, 999)}',
            'start_time': start_time.isoformat(),
            'duration': duration,
            'country': np.random.choice(countries),
            'user_agent': np.random.choice(user_agents),
            'device': np.random.choice(['desktop', 'mobile', 'tablet']),
            'converted': np.random.random() > 0.7,
            'error_count': np.random.randint(0, 5) if has_error else 0,
            'page_views': np.random.randint(1, 15),
            'bounce': duration < 10 and np.random.random() > 0.3
        })
    
    return sessions


# Generate all data
time_series_data, error_windows = generate_time_series_data()
splunk_logs = generate_splunk_logs(error_windows)
topology_data = generate_topology_data()
problems_data = generate_problems_data()
user_sessions_data = generate_user_sessions()

# Save logs to file when the app starts
save_logs_to_file(splunk_logs)

# Convert to DataFrames for easier display
problems_df = pd.DataFrame(problems_data)
user_sessions_df = pd.DataFrame(user_sessions_data)
splunk_logs_df = pd.DataFrame(splunk_logs)

# Streamlit app layout
st.title("📊 Dynatrace Synthetic Data Visualization")
st.markdown("This dashboard displays synthetic data mimicking Dynatrace metrics and Splunk logs for testing visualization applications.")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    selected_service = st.selectbox(
        "Select Service",
        options=["All"] + list(time_series_data['service'].unique())
    )
    
    time_range = st.slider(
        "Time Range (hours)",
        min_value=1,
        max_value=24,
        value=12
    )
    
    st.markdown("---")
    st.markdown("**Visualization Options**")
    show_topology = st.checkbox("Show Service Topology", value=True)
    show_problems = st.checkbox("Show Problems", value=True)
    show_user_sessions = st.checkbox("Show User Sessions", value=True)
    show_logs = st.checkbox("Show Splunk Logs", value=True)
    
    # Log download button
    if st.button("Download Logs as TXT"):
        save_logs_to_file(splunk_logs)

# Filter time series data based on selections
filtered_time_series = time_series_data[
    time_series_data['timestamp'] >= (datetime.now() - timedelta(hours=time_range))
]

if selected_service != "All":
    filtered_time_series = filtered_time_series[filtered_time_series['service'] == selected_service]

# ... [keep all the existing visualization code until the Splunk Logs section] ...

# Enhanced Splunk Logs display
if show_logs:
    st.subheader("Splunk/Kibana-style Logs")
    
    # Filter logs based on time range
    filtered_logs = [log for log in splunk_logs 
                    if datetime.fromisoformat(log['timestamp']) >= (datetime.now() - timedelta(hours=time_range))]
    
    # Add filters for log inspection
    col1, col2 = st.columns(2)
    with col1:
        log_level_filter = st.multiselect(
            "Filter by log level",
            options=["INFO", "WARN", "ERROR", "DEBUG"],
            default=["ERROR"]
        )
    
    with col2:
        error_type_filter = st.multiselect(
            "Filter by error type",
            options=list(set([p["type"] for p in ERROR_PATTERNS])),
            default=[p["type"] for p in ERROR_PATTERNS]
        )
    
    # Apply filters
    filtered_logs = [
        log for log in filtered_logs 
        if log['level'] in log_level_filter and 
        (log.get('error_type') in error_type_filter if 'error_type' in log else True)
    ]
    
    # Show log statistics
    st.markdown("**Log Statistics**")
    if filtered_logs:
        log_stats = pd.DataFrame(filtered_logs)
        
        if 'timestamp' in log_stats.columns:
            log_stats['timestamp'] = pd.to_datetime(log_stats['timestamp'])
            log_stats['time'] = log_stats['timestamp'].dt.floor('H')
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Logs by Hour**")
                hourly_counts = log_stats.groupby('time').size().reset_index(name='count')
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=hourly_counts, x='time', y='count', ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Error Types**")
                if 'error_type' in log_stats.columns:
                    error_counts = log_stats['error_type'].value_counts().reset_index()
                    error_counts.columns = ['error_type', 'count']
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=error_counts, x='error_type', y='count', ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("No error type information available for selected logs")
    
    # Show detailed logs with expandable view
    st.markdown("**Detailed Logs**")
    if filtered_logs:
        # Add button to inspect logs during spikes
        if st.button("Show Logs During Error Spikes"):
            spike_logs = []
            for window_start, window_end in error_windows:
                if window_start >= (datetime.now() - timedelta(hours=time_range)):
                    spike_logs.extend([
                        log for log in filtered_logs 
                        if window_start <= datetime.fromisoformat(log['timestamp']) <= window_end
                    ])
            
            if spike_logs:
                st.json(spike_logs[-50:], expanded=False)
            else:
                st.warning("No logs found during error spikes in the selected time range")
        
        # Show all filtered logs
        st.json(filtered_logs[-100:], expanded=False)
        
        # Add button to save filtered logs
        if st.button("Save Filtered Logs"):
            save_logs_to_file(filtered_logs, "filtered_logs.txt")
    else:
        st.info("No logs match the current filters")

# Filter time series data based on selections
filtered_time_series = time_series_data[
    time_series_data['timestamp'] >= (datetime.now() - timedelta(hours=time_range))
]

if selected_service != "All":
    filtered_time_series = filtered_time_series[filtered_time_series['service'] == selected_service]

# Metrics overview
st.subheader("Key Metrics Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg CPU Usage", f"{filtered_time_series['cpu_usage'].mean():.1f}%")
col2.metric("Avg Memory Usage", f"{filtered_time_series['memory_usage'].mean():.1f}%")
col3.metric("Avg Response Time", f"{filtered_time_series['response_time'].mean():.1f} ms")
col4.metric("Avg Error Rate", f"{filtered_time_series['error_rate'].mean():.2f}%")

# Time series charts
st.subheader("Time Series Metrics")
tab1, tab2, tab3, tab4 = st.tabs(["CPU Usage", "Memory Usage", "Response Time", "Error Rate"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_time_series, x='timestamp', y='cpu_usage', hue='service', ax=ax)
    
    # Highlight error windows
    for start, end in error_windows:
        if start >= (datetime.now() - timedelta(hours=time_range)):
            ax.axvspan(start, end, color='red', alpha=0.1)
    
    ax.set_title("CPU Usage Over Time (Red areas = error periods)")
    ax.set_ylabel("CPU Usage (%)")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_time_series, x='timestamp', y='memory_usage', hue='service', ax=ax)
    
    # Highlight error windows
    for start, end in error_windows:
        if start >= (datetime.now() - timedelta(hours=time_range)):
            ax.axvspan(start, end, color='red', alpha=0.1)
    
    ax.set_title("Memory Usage Over Time (Red areas = error periods)")
    ax.set_ylabel("Memory Usage (%)")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_time_series, x='timestamp', y='response_time', hue='service', ax=ax)
    
    # Highlight error windows
    for start, end in error_windows:
        if start >= (datetime.now() - timedelta(hours=time_range)):
            ax.axvspan(start, end, color='red', alpha=0.1)
    
    ax.set_title("Response Time Over Time (Red areas = error periods)")
    ax.set_ylabel("Response Time (ms)")
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_time_series, x='timestamp', y='error_rate', hue='service', ax=ax)
    
    # Highlight error windows
    for start, end in error_windows:
        if start >= (datetime.now() - timedelta(hours=time_range)):
            ax.axvspan(start, end, color='red', alpha=0.1)
    
    ax.set_title("Error Rate Over Time (Red areas = error periods)")
    ax.set_ylabel("Error Rate (%)")
    st.pyplot(fig)

# Service Topology (keep the same as before)
if show_topology:
    st.subheader("Service Topology")
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in topology_data['nodes']:
        G.add_node(node['id'], name=node['name'], type=node['type'], health=node['health'])
    
    # Add edges
    for edge in topology_data['edges']:
        G.add_edge(edge['from'], edge['to'], 
                  weight=edge['request_count'], 
                  error_rate=edge['error_rate'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Node colors based on health
    node_colors = ['green' if node['health'] == 'HEALTHY' else 'red' for node in topology_data['nodes']]
    
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['name'] for n in G.nodes()}, ax=ax)
    
    # Edge widths based on request count
    edge_widths = [d['weight']/200 for u, v, d in G.edges(data=True)]
    edge_colors = ['red' if d['error_rate'] > 2 else 'gray' for u, v, d in G.edges(data=True)]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=True, ax=ax)
    
    # Add legend
    healthy_patch = plt.Line2D([0], [0], marker='o', color='w', label='Healthy',
                             markerfacecolor='green', markersize=10)
    unhealthy_patch = plt.Line2D([0], [0], marker='o', color='w', label='Unhealthy',
                               markerfacecolor='red', markersize=10)
    plt.legend(handles=[healthy_patch, unhealthy_patch])
    
    plt.title("Service Topology (Node size = health, Edge width = traffic, Red edges = high errors)")
    st.pyplot(fig)

# Problems table (keep the same as before)
if show_problems:
    st.subheader("Recent Problems")
    
    # Color coding for severity
    def color_severity(severity):
        if severity == 'AVAILABILITY':
            return 'background-color: #ff0000; color: white'
        elif severity == 'ERROR':
            return 'background-color: #ff6600; color: white'
        elif severity == 'PERFORMANCE':
            return 'background-color: #ffcc00; color: black'
        else:
            return 'background-color: #0066cc; color: white'
    
    styled_problems = problems_df.style.applymap(
        lambda x: color_severity(x) if pd.notna(x) else '', 
        subset=['severity']
    )
    
    st.dataframe(
        styled_problems,
        column_order=['problem_id', 'title', 'severity', 'status', 'start_time', 'affected_services', 'root_cause'],
        use_container_width=True
    )

# User sessions (keep the same as before)
if show_user_sessions:
    st.subheader("User Session Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Session Duration Distribution**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=user_sessions_df, x='duration', bins=20, ax=ax)
        ax.set_xlabel("Duration (seconds)")
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Conversion Rate by Device**")
        conv_rates = user_sessions_df.groupby('device')['converted'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=conv_rates, x='device', y='converted', ax=ax)
        ax.set_ylabel("Conversion Rate")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    
    st.markdown("**User Sessions by Country**")
    country_counts = user_sessions_df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'sessions']
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=country_counts, x='country', y='sessions', ax=ax)
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of Sessions")
    st.pyplot(fig)

# Splunk Logs display
if show_logs:
    st.subheader("Splunk/Kibana-style Logs")
    
    # Filter logs based on time range
    filtered_logs = [log for log in splunk_logs 
                    if datetime.fromisoformat(log['timestamp']) >= (datetime.now() - timedelta(hours=time_range))]
    
    # Show log count by level
    st.markdown("**Log Level Distribution**")
    log_levels = pd.DataFrame(filtered_logs)['level'].value_counts().reset_index()
    log_levels.columns = ['level', 'count']
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=log_levels, x='level', y='count', ax=ax)
    st.pyplot(fig)
    
    # Show error logs
    error_logs = [log for log in filtered_logs if log['level'] == 'ERROR']
    if error_logs:
        st.markdown("**Error Logs (Last 20)**")
        st.json(error_logs[-20:], expanded=False)
    else:
        st.info("No error logs in the selected time range")
    
    # Show raw logs option
    if st.checkbox("Show raw logs sample (Last 50)"):
        st.json(filtered_logs[-50:], expanded=False)

# Add some space at the bottom
st.markdown("---")
st.caption("Synthetic data generated for Dynatrace and Splunk visualization testing")
