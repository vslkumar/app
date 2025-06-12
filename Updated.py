import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Custom CSS for improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f2f6, #ffffff);
    }
    
    .stSelectbox > div > div > div {
        background-color: white;
        border-radius: 5px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .status-healthy {
        color: #2ca02c;
        font-weight: bold;
    }
    
    .status-unhealthy {
        color: #d62728;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ff7f0e;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def save_logs_to_file(logs, filename="logs.txt"):
    """Save logs to a text file in JSON format"""
    with open(filename, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')
    st.sidebar.success(f"✅ Logs saved to {filename}")

# Error patterns to inject 4 times a day
ERROR_PATTERNS = [
    {"type": "Request Timeout", "codes": ["504", "ETIMEDOUT"], "impact": ["response_time", "error_rate"]},
    {"type": "SSL/TLS Errors", "codes": ["ERR_CERT_AUTHORITY_INVALID", "SSL_ERROR"], "impact": ["error_rate"]},
    {"type": "Connection Refused", "codes": ["ECONNREFUSED"], "impact": ["error_rate"]},
    {"type": "4xx Errors", "codes": ["400", "401", "403", "404", "405", "409", "413", "429"], "impact": ["error_rate"]},
    {"type": "5xx Errors", "codes": ["500", "502", "503", "504"], "impact": ["response_time", "error_rate"]}
]

# Professional color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Generate synthetic data functions with error injection
@st.cache_data(ttl=300)
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

@st.cache_data(ttl=300)
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
            "status_code": str(random.choices([200, 201, 400, 401, 403, 404, 500, 502, 503], 
                                         weights=[0.4, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05])[0]),
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
                "status_code": str(status_code),
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

@st.cache_data(ttl=300)
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

@st.cache_data(ttl=300)
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

@st.cache_data(ttl=300)
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

def create_time_series_chart(data, metric, title, color=COLORS['primary']):
    """Create an interactive time series chart with Plotly"""
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data[metric],
        mode='lines+markers',
        name=metric.replace('_', ' ').title(),
        line=dict(color=color, width=2),
        marker=dict(size=4, opacity=0.8),
        hovertemplate='<b>%{y:.2f}</b><br>%{x}<br><extra></extra>'
    ))
    
    # Add trend line
    z = np.polyfit(range(len(data)), data[metric], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=p(range(len(data))),
        mode='lines',
        name='Trend',
        line=dict(color='rgba(255,127,14,0.5)', width=1, dash='dash'),
        hovertemplate='Trend: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, weight='bold')),
        xaxis_title="Time",
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_service_topology_chart(topology_data):
    """Create an interactive service topology visualization"""
    G = nx.DiGraph()
    
    # Add nodes
    for service in topology_data['nodes']:
        G.add_node(service['id'], **service)
    
    # Add edges
    for rel in topology_data['edges']:
        G.add_edge(rel['from'], rel['to'], **rel)
    
    # Generate layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        edge_data = G.edges[edge]
        edge_info.append(f"Requests: {edge_data['request_count']}<br>Error Rate: {edge_data['error_rate']:.1f}%")
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        node_text.append(f"<b>{node_data['name']}</b><br>Type: {node_data['type']}<br>Health: {node_data['health']}")
        
        # Color based on health
        if node_data['health'] == 'HEALTHY':
            node_color.append(COLORS['success'])
        elif node_data['health'] == 'UNHEALTHY':
            node_color.append(COLORS['danger'])
        else:
            node_color.append(COLORS['warning'])
        
        # Size based on type
        if node_data['type'] == 'APPLICATION':
            node_size.append(30)
        elif node_data['type'] == 'DATABASE':
            node_size.append(25)
        else:
            node_size.append(20)
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
        text=[G.nodes[node]['name'] for node in G.nodes()],
        textposition="bottom center",
        hovertemplate='%{text}<extra></extra>',
        hovertext=node_text,
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text="Service Topology", x=0.5, font=dict(size=18, weight='bold')),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="Node size indicates service type • Color indicates health status",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=10)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=500
    )
    
    return fig

def create_problems_timeline(problems_data):
    """Create a timeline visualization for problems"""
    df = pd.DataFrame(problems_data)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # Create timeline chart
    fig = go.Figure()
    
    colors = {
        'AVAILABILITY': COLORS['danger'],
        'ERROR': COLORS['warning'],
        'PERFORMANCE': COLORS['info'],
        'RESOURCE': COLORS['secondary']
    }
    
    for i, problem in df.iterrows():
        end_time = problem['end_time'] if pd.notna(problem['end_time']) else datetime.now()
        duration = (end_time - problem['start_time']).total_seconds() / 3600  # hours
        
        fig.add_trace(go.Scatter(
            x=[problem['start_time'], end_time],
            y=[i, i],
            mode='lines+markers',
            line=dict(color=colors.get(problem['severity'], COLORS['primary']), width=8),
            marker=dict(size=8),
            name=problem['severity'],
            showlegend=problem['severity'] not in [trace.name for trace in fig.data],
            hovertemplate=f"<b>{problem['title']}</b><br>" +
                         f"Severity: {problem['severity']}<br>" +
                         f"Status: {problem['status']}<br>" +
                         f"Duration: {duration:.1f}h<br>" +
                         f"Services: {', '.join(problem['affected_services'])}<br>" +
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="Problems Timeline", x=0.5, font=dict(size=18, weight='bold')),
        xaxis_title="Time",
        yaxis_title="Problems",
        yaxis=dict(tickmode='array', tickvals=list(range(len(df))), ticktext=df['problem_id']),
        height=400,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

def create_user_sessions_chart(sessions_data):
    """Create user sessions analytics charts"""
    df = pd.DataFrame(sessions_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sessions by Country', 'Device Distribution', 'Conversion Rate', 'Session Duration'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Sessions by country
    country_counts = df['country'].value_counts()
    fig.add_trace(
        go.Bar(x=country_counts.index, y=country_counts.values, 
               marker_color=COLORS['primary'], name='Sessions by Country'),
        row=1, col=1
    )
    
    # Device distribution
    device_counts = df['device'].value_counts()
    fig.add_trace(
        go.Pie(labels=device_counts.index, values=device_counts.values, 
               name='Device Distribution'),
        row=1, col=2
    )
    
    # Conversion rate by device
    conversion_by_device = df.groupby('device')['converted'].mean()
    fig.add_trace(
        go.Bar(x=conversion_by_device.index, y=conversion_by_device.values,
               marker_color=COLORS['success'], name='Conversion Rate'),
        row=2, col=1
    )
    
    # Session duration histogram
    fig.add_trace(
        go.Histogram(x=df['duration'], nbinsx=20, 
                    marker_color=COLORS['info'], name='Session Duration'),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(text="User Sessions Analytics", x=0.5, font=dict(size=18, weight='bold')),
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

# Initialize data
with st.spinner('Loading dashboard data...'):
    time_series_data, error_windows = generate_time_series_data()
    splunk_logs = generate_splunk_logs(error_windows)
    topology_data = generate_topology_data()
    problems_data = generate_problems_data()
    user_sessions_data = generate_user_sessions()

# Convert to DataFrames
problems_df = pd.DataFrame(problems_data)
user_sessions_df = pd.DataFrame(user_sessions_data)
splunk_logs_df = pd.DataFrame(splunk_logs)

# Main header
st.markdown('<h1 class="main-header">📊 Dynatrace Data Visualization Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Real-time monitoring and analytics for your infrastructure")

# Sidebar filters
with st.sidebar:
    st.header("🔧 Dashboard Controls")
    
    # Service selection
    selected_service = st.selectbox(
        "🎯 Select Service",
        options=["All"] + list(time_series_data['service'].unique()),
        help="Filter metrics by specific service"
    )
    
    # Time range
    time_range = st.slider(
        "⏰ Time Range (hours)",
        min_value=1,
        max_value=24,
        value=12,
        help="Adjust the time window for analysis"
    )
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()
    
    st.markdown("---")
    
    # Visualization toggles
    st.markdown("### 📊 Visualization Options")
    show_topology = st.checkbox("🔗 Service Topology", value=True)
    show_problems = st.checkbox("⚠️ Problems Timeline", value=True)
    show_user_sessions = st.checkbox("👥 User Sessions", value=True)
    show_logs = st.checkbox("📋 Splunk Logs", value=True)
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 📤 Export Data")
    if st.button("💾 Download Logs", use_container_width=True):
        save_logs_to_file(splunk_logs)
    
    if st.button("📊 Export Metrics", use_container_width=True):
        csv = time_series_data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="metrics_data.csv",
            mime="text/csv"
        )

# Filter data based on selections
current_time = datetime.now()
cutoff_time = current_time - timedelta(hours=time_range)
filtered_time_series = time_series_data[time_series_data['timestamp'] >= cutoff_time]

if selected_service != "All":
    filtered_time_series = filtered_time_series[filtered_time_series['service'] == selected_service]

# Key metrics cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_cpu = filtered_time_series['cpu_usage'].mean()
    st.metric(
        label="🖥️ Avg CPU Usage", 
        value=f"{avg_cpu:.1f}%",
        delta=f"{avg_cpu - 30:.1f}%"
    )

with col2:
    avg_memory = filtered_time_series['memory_usage'].mean()
    st.metric(
        label="💾 Avg Memory Usage", 
        value=f"{avg_memory:.1f}%",
        delta=f"{avg_memory - 45:.1f}%"
    )

with col3:
    avg_response = filtered_time_series['response_time'].mean()
    st.metric(
        label="⚡ Avg Response Time", 
        value=f"{avg_response:.0f}ms",
        delta=f"{avg_response - 50:.0f}ms"
    )

with col4:
    avg_error = filtered_time_series['error_rate'].mean()
    st.metric(
        label="❌ Avg Error Rate", 
        value=f"{avg_error:.2f}%",
        delta=f"{avg_error - 0.5:.2f}%"
    )

# Time series charts
st.markdown("## 📈 Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    cpu_chart = create_time_series_chart(filtered_time_series, 'cpu_usage', 'CPU Usage Over Time', COLORS['primary'])
    st.plotly_chart(cpu_chart, use_container_width=True)
    
    # CPU Analysis
    st.markdown("### 🔍 CPU Usage Analysis")
    cpu_max = filtered_time_series['cpu_usage'].max()
    cpu_min = filtered_time_series['cpu_usage'].min()
    cpu_std = filtered_time_series['cpu_usage'].std()
    cpu_trend = "increasing" if filtered_time_series['cpu_usage'].iloc[-10:].mean() > filtered_time_series['cpu_usage'].iloc[:10].mean() else "decreasing"
    
    st.markdown(f"""
    <div class="metric-card">
        <p><strong>Peak Usage:</strong> {cpu_max:.1f}%</p>
        <p><strong>Minimum Usage:</strong> {cpu_min:.1f}%</p>
        <p><strong>Variability:</strong> {cpu_std:.1f}% (std dev)</p>
        <p><strong>Trend:</strong> {cpu_trend.capitalize()}</p>
        <p><strong>Status:</strong> {'⚠️ High' if cpu_max > 80 else '✅ Normal'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    response_chart = create_time_series_chart(filtered_time_series, 'response_time', 'Response Time Over Time', COLORS['warning'])
    st.plotly_chart(response_chart, use_container_width=True)
    
    # Response Time Analysis
    st.markdown("### 🔍 Response Time Analysis")
    resp_max = filtered_time_series['response_time'].max()
    resp_avg = filtered_time_series['response_time'].mean()
    resp_p95 = filtered_time_series['response_time'].quantile(0.95)
    slow_requests = (filtered_time_series['response_time'] > 500).sum()
    
    st.markdown(f"""
    <div class="metric-card">
        <p><strong>Max Response Time:</strong> {resp_max:.0f}ms</p>
        <p><strong>95th Percentile:</strong> {resp_p95:.0f}ms</p>
        <p><strong>Average:</strong> {resp_avg:.0f}ms</p>
        <p><strong>Slow Requests (>500ms):</strong> {slow_requests}</p>
        <p><strong>Performance:</strong> {'❌ Poor' if resp_p95 > 1000 else '⚠️ Degraded' if resp_p95 > 500 else '✅ Good'}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    memory_chart = create_time_series_chart(filtered_time_series, 'memory_usage', 'Memory Usage Over Time', COLORS['success'])
    st.plotly_chart(memory_chart, use_container_width=True)
    
    # Memory Analysis
    st.markdown("### 🔍 Memory Usage Analysis")
    mem_max = filtered_time_series['memory_usage'].max()
    mem_min = filtered_time_series['memory_usage'].min()
    mem_growth = filtered_time_series['memory_usage'].iloc[-1] - filtered_time_series['memory_usage'].iloc[0]
    mem_trend = "increasing" if mem_growth > 5 else "stable" if abs(mem_growth) <= 5 else "decreasing"
    
    st.markdown(f"""
    <div class="metric-card">
        <p><strong>Peak Usage:</strong> {mem_max:.1f}%</p>
        <p><strong>Minimum Usage:</strong> {mem_min:.1f}%</p>
        <p><strong>Growth:</strong> {mem_growth:+.1f}% over period</p>
        <p><strong>Trend:</strong> {mem_trend.capitalize()}</p>
        <p><strong>Status:</strong> {'❌ Critical' if mem_max > 90 else '⚠️ High' if mem_max > 75 else '✅ Normal'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    error_chart = create_time_series_chart(filtered_time_series, 'error_rate', 'Error Rate Over Time', COLORS['danger'])
    st.plotly_chart(error_chart, use_container_width=True)
    
    # Error Rate Analysis
    st.markdown("### 🔍 Error Rate Analysis")
    error_max = filtered_time_series['error_rate'].max()
    error_avg = filtered_time_series['error_rate'].mean()
    error_spikes = (filtered_time_series['error_rate'] > 5).sum()
    error_incidents = (filtered_time_series['error_rate'] > 10).sum()
    
    st.markdown(f"""
    <div class="metric-card">
        <p><strong>Peak Error Rate:</strong> {error_max:.2f}%</p>
        <p><strong>Average Error Rate:</strong> {error_avg:.2f}%</p>
        <p><strong>Error Spikes (>5%):</strong> {error_spikes}</p>
        <p><strong>Critical Incidents (>10%):</strong> {error_incidents}</p>
        <p><strong>Reliability:</strong> {'❌ Poor' if error_avg > 5 else '⚠️ Degraded' if error_avg > 2 else '✅ Excellent'}</p>
    </div>
    """, unsafe_allow_html=True)

# Service topology
if show_topology:
    st.markdown("## 🔗 Service Topology")
    topology_chart = create_service_topology_chart(topology_data)
    st.plotly_chart(topology_chart, use_container_width=True)
    
    # Service health summary
    st.markdown("### 🏥 Service Health Summary")
    health_cols = st.columns(len(topology_data['nodes']))
    
    for i, service in enumerate(topology_data['nodes']):
        with health_cols[i]:
            health_color = "status-healthy" if service['health'] == 'HEALTHY' else "status-unhealthy"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{service['name']}</h4>
                <p><strong>Type:</strong> {service['type']}</p>
                <p><strong>Status:</strong> <span class="{health_color}">{service['health']}</span></p>
            </div>
            """, unsafe_allow_html=True)

# Problems timeline
if show_problems:
    st.markdown("## ⚠️ Problems Timeline")
    problems_chart = create_problems_timeline(problems_data)
    st.plotly_chart(problems_chart, use_container_width=True)
    
    # Problems summary table
    st.markdown("### 📋 Recent Problems")
    problems_display = problems_df[['problem_id', 'title', 'severity', 'status', 'start_time']].copy()
    problems_display['start_time'] = pd.to_datetime(problems_display['start_time']).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(problems_display, use_container_width=True)

# User sessions analytics
if show_user_sessions:
    st.markdown("## 👥 User Sessions Analytics")
    sessions_chart = create_user_sessions_chart(user_sessions_data)
    st.plotly_chart(sessions_chart, use_container_width=True)
    
    # Sessions summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sessions = len(user_sessions_data)
        st.metric("Total Sessions", total_sessions)
    
    with col2:
        avg_duration = user_sessions_df['duration'].mean()
        st.metric("Avg Duration", f"{avg_duration:.0f}s")
    
    with col3:
        conversion_rate = user_sessions_df['converted'].mean() * 100
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    
    with col4:
        bounce_rate = user_sessions_df['bounce'].mean() * 100
        st.metric("Bounce Rate", f"{bounce_rate:.1f}%")

# Splunk logs analysis
if show_logs:
    st.markdown("## 📋 Splunk Logs Analysis")
    
    # Log level distribution
    log_levels = splunk_logs_df['level'].value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig_pie = px.pie(
            values=log_levels.values, 
            names=log_levels.index, 
            title="Log Level Distribution",
            color_discrete_map={
                'INFO': COLORS['info'],
                'WARN': COLORS['warning'],
                'ERROR': COLORS['danger'],
                'DEBUG': COLORS['light']
            }
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Error logs over time
        error_logs = splunk_logs_df[splunk_logs_df['level'] == 'ERROR'].copy()
        if not error_logs.empty:
            error_logs['timestamp'] = pd.to_datetime(error_logs['timestamp'])
            error_logs['hour'] = error_logs['timestamp'].dt.floor('h')
            error_timeline = error_logs.groupby('hour').size().reset_index(name='count')
            
            fig_timeline = px.bar(
                error_timeline, 
                x='hour', 
                y='count', 
                title="Error Logs Timeline",
                color_discrete_sequence=[COLORS['danger']]
            )
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Recent logs table
    st.markdown("### 📄 Recent Logs")
    recent_logs = splunk_logs_df.head(100)[['timestamp', 'service', 'level', 'message', 'status_code']].copy()
    recent_logs['timestamp'] = pd.to_datetime(recent_logs['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add search functionality
    search_term = st.text_input("🔍 Search logs", placeholder="Enter search term...")
    if search_term:
        recent_logs = recent_logs[recent_logs['message'].str.contains(search_term, case=False, na=False)]
    
    st.dataframe(recent_logs, use_container_width=True, height=400)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🔄 Dashboard auto-refreshes every 5 minutes | 📊 Data generated synthetically for demo purposes</p>
        <p>Built with ❤️ using Streamlit and Plotly</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Auto-refresh mechanism
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
