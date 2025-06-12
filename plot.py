import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Dynatrace Data Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Generate synthetic data functions
def generate_time_series_data():
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    timestamps = pd.date_range(start_time, end_time, freq='5min')
    
    cpu_base = np.random.normal(30, 5, len(timestamps))
    cpu_spikes = np.random.choice([0, 1], size=len(timestamps), p=[0.95, 0.05])
    cpu_usage = np.clip(cpu_base + cpu_spikes[0] * np.random.uniform(20, 50), 0, 100)
    
    memory_usage = np.clip(np.random.normal(45, 8, len(timestamps)), 0, 100)
    response_time = np.clip(np.random.gamma(2, 10, len(timestamps)), 5, 500)
    error_rate = np.clip(np.random.exponential(0.5, len(timestamps)), 0, 10)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,  # Fixed column name with underscore
        'response_time': response_time,
        'error_rate': error_rate,
        'service': ['frontend' if i % 2 == 0 else 'backend' for i in range(len(timestamps))]
    })
    
    return df

def generate_topology_data():
    services = [
        {'id': 'frontend', 'name': 'Web Frontend', 'type': 'APPLICATION', 'health': 'HEALTHY'},
        {'id': 'backend', 'name': 'Backend Service', 'type': 'SERVICE', 'health': 'HEALTHY'},
        {'id': 'database', 'name': 'Customer DB', 'type': 'DATABASE', 'health': 'HEALTHY'},
        {'id': 'payment', 'name': 'Payment Service', 'type': 'SERVICE', 'health': 'UNHEALTHY'},
        {'id': 'auth', 'name': 'Auth Service', 'type': 'SERVICE', 'health': 'HEALTHY'}
    ]
    
    relationships = [
        {'from': 'frontend', 'to': 'backend', 'request_count': 1200, 'error_rate': 0.5},
        {'from': 'backend', 'to': 'database', 'request_count': 2400, 'error_rate': 0.2},
        {'from': 'backend', 'to': 'payment', 'request_count': 800, 'error_rate': 5.2},
        {'from': 'backend', 'to': 'auth', 'request_count': 1500, 'error_rate': 1.1},
        {'from': 'frontend', 'to': 'auth', 'request_count': 300, 'error_rate': 0.3}
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
            'title': f'High response time in {np.random.choice(["frontend", "backend", "payment"])}',
            'severity': np.random.choice(severities),
            'status': 'OPEN' if end_time is None else 'CLOSED',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat() if end_time else None,
            'affected_services': np.random.choice(['frontend', 'backend', 'database', 'payment', 'auth'], 
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
time_series_data = generate_time_series_data()
topology_data = generate_topology_data()
problems_data = generate_problems_data()
user_sessions_data = generate_user_sessions()

# Convert to DataFrames for easier display
problems_df = pd.DataFrame(problems_data)
user_sessions_df = pd.DataFrame(user_sessions_data)

# Streamlit app layout
st.title("📊 Dynatrace Synthetic Data Visualization")
st.markdown("This dashboard displays synthetic data mimicking Dynatrace metrics for testing visualization applications.")

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
col2.metric("Avg Memory Usage", f"{filtered_time_series['memory_usage'].mean():.1f}%")  # Fixed column name
col3.metric("Avg Response Time", f"{filtered_time_series['response_time'].mean():.1f} ms")
col4.metric("Avg Error Rate", f"{filtered_time_series['error_rate'].mean():.2f}%")

# Time series charts
st.subheader("Time Series Metrics")
tab1, tab2, tab3, tab4 = st.tabs(["CPU Usage", "Memory Usage", "Response Time", "Error Rate"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_time_series, x='timestamp', y='cpu_usage', hue='service', ax=ax)
    ax.set_title("CPU Usage Over Time")
    ax.set_ylabel("CPU Usage (%)")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_time_series, x='timestamp', y='memory_usage', hue='service', ax=ax)  # Fixed column name
    ax.set_title("Memory Usage Over Time")
    ax.set_ylabel("Memory Usage (%)")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_time_series, x='timestamp', y='response_time', hue='service', ax=ax)
    ax.set_title("Response Time Over Time")
    ax.set_ylabel("Response Time (ms)")
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=filtered_time_series, x='timestamp', y='error_rate', hue='service', ax=ax)
    ax.set_title("Error Rate Over Time")
    ax.set_ylabel("Error Rate (%)")
    st.pyplot(fig)

# Service Topology
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

# Problems table
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

# User sessions
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

# Add some space at the bottom
st.markdown("---")
st.caption("Synthetic data generated for Dynatrace visualization testing")
