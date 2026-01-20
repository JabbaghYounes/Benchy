# HTML Dashboard Generator for Benchmark Results
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from benchmark.aggregation import (
    ResultsAggregator,
    YOLOAggregatedMetrics,
    LLMAggregatedMetrics,
    PlatformSummary,
)

logger = logging.getLogger(__name__)


# Color palette for platforms (neutral, distinct colors)
PLATFORM_COLORS = {
    "jetson_nano": "#2563eb",       # Blue
    "rpi_ai_hat_plus": "#16a34a",   # Green
    "rpi_ai_hat_plus_2": "#dc2626", # Red
}

# Chart.js CDN
CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


class DashboardGenerator:
    """Generates static HTML dashboard for benchmark results."""

    def __init__(self, aggregator: ResultsAggregator):
        """Initialize dashboard generator.

        Args:
            aggregator: ResultsAggregator with loaded data
        """
        self.aggregator = aggregator
        self.platform_summaries = aggregator.get_platform_summaries()
        self.yolo_metrics = aggregator.aggregate_yolo_results()
        self.llm_metrics = aggregator.aggregate_llm_results()

    def generate(
        self,
        output_path: Path,
        title: str = "Edge AI Benchmark Dashboard",
        include_raw_data: bool = True,
    ) -> None:
        """Generate the HTML dashboard.

        Args:
            output_path: Path to write the HTML file
            title: Dashboard title
            include_raw_data: Include raw data tables
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._generate_html(title, include_raw_data)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Dashboard written to: {output_path}")

    def _generate_html(self, title: str, include_raw_data: bool) -> str:
        """Generate complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="{CHARTJS_CDN}"></script>
    {self._generate_styles()}
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        {self._generate_filters_section()}
        {self._generate_system_overview_section()}
        {self._generate_yolo_performance_section()}
        {self._generate_yolo_scaling_section()}
        {self._generate_llm_performance_section()}
        {self._generate_llm_efficiency_section()}
        {self._generate_variance_section()}
        {self._generate_raw_data_section() if include_raw_data else ""}
        {self._generate_download_section()}

        <footer>
            <p>Edge AI Benchmark Suite - <a href="https://github.com/edge-ai-benchmark">Documentation</a></p>
        </footer>
    </div>

    {self._generate_scripts()}
</body>
</html>"""

    def _generate_styles(self) -> str:
        """Generate CSS styles."""
        return """
    <style>
        :root {
            --primary: #1e40af;
            --secondary: #64748b;
            --success: #16a34a;
            --warning: #ca8a04;
            --danger: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border);
        }

        header h1 {
            font-size: 2rem;
            color: var(--primary);
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        section {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        section h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        section h3 {
            font-size: 1rem;
            margin: 1rem 0 0.5rem;
            color: var(--text);
        }

        .grid {
            display: grid;
            gap: 1.5rem;
        }

        .grid-2 { grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }
        .grid-3 { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
        .grid-4 { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }

        .card {
            background: var(--bg);
            border-radius: 6px;
            padding: 1rem;
        }

        .stat-card {
            text-align: center;
            padding: 1.5rem;
        }

        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }

        .stat-card .label {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }

        .chart-container.tall {
            height: 400px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg);
            font-weight: 600;
            color: var(--text-muted);
        }

        tr:hover {
            background: var(--bg);
        }

        .filters {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .filter-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .filter-group label {
            font-weight: 500;
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        select, button {
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: white;
            font-size: 0.9rem;
            cursor: pointer;
        }

        button {
            background: var(--primary);
            color: white;
            border: none;
        }

        button:hover {
            opacity: 0.9;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        .badge-blue { background: #dbeafe; color: #1e40af; }
        .badge-green { background: #dcfce7; color: #166534; }
        .badge-red { background: #fee2e2; color: #991b1b; }
        .badge-purple { background: #ede9fe; color: #5b21b6; }
        .badge-orange { background: #ffedd5; color: #9a3412; }

        .download-links {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .download-links a {
            padding: 0.5rem 1rem;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            text-decoration: none;
            color: var(--text);
            font-size: 0.85rem;
        }

        .download-links a:hover {
            background: var(--border);
        }

        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        footer a {
            color: var(--primary);
        }

        .platform-legend {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 1rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }

        .no-data {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
        }

        @media (max-width: 768px) {
            .container { padding: 1rem; }
            .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
            .chart-container { height: 250px; }
        }
    </style>"""

    def _generate_filters_section(self) -> str:
        """Generate global filters section.

        Model Expansion PRD - Phase 6 Task 6.2: Add 1B/3B selectors, MoE/dense/code filters.
        """
        platforms = [s.platform for s in self.platform_summaries]
        yolo_versions = list(set(m.yolo_version for m in self.yolo_metrics))
        tasks = list(set(m.task for m in self.yolo_metrics))
        llm_sizes = list(set(m.model_size for m in self.llm_metrics))

        # Phase 6: Extract new filter dimensions
        llm_param_groups = list(set(
            m.parameter_group for m in self.llm_metrics
            if m.parameter_group
        ))
        llm_architectures = list(set(
            m.architecture for m in self.llm_metrics
            if m.architecture
        ))
        llm_specializations = list(set(
            m.specialization for m in self.llm_metrics
            if m.specialization
        ))

        platform_options = "".join(f'<option value="{p}">{p.replace("_", " ").title()}</option>' for p in platforms)
        version_options = "".join(f'<option value="{v}">{v}</option>' for v in sorted(yolo_versions))
        task_options = "".join(f'<option value="{t}">{t.title()}</option>' for t in sorted(tasks))
        size_options = "".join(f'<option value="{s}">{s}</option>' for s in sorted(llm_sizes))

        # Phase 6: New filter options
        param_group_options = "".join(
            f'<option value="{pg}">{pg}</option>'
            for pg in sorted(llm_param_groups, key=lambda x: int(x.rstrip('B')) if x.rstrip('B').isdigit() else 999)
        )
        arch_options = "".join(
            f'<option value="{a}">{a.upper() if a == "moe" else a.title()}</option>'
            for a in sorted(llm_architectures)
        )
        spec_options = "".join(
            f'<option value="{s}">{s.title()}</option>'
            for s in sorted(llm_specializations)
        )

        return f"""
        <section id="filters">
            <h2>Filters</h2>
            <div class="filters">
                <div class="filter-group">
                    <label>Platform:</label>
                    <select id="filter-platform" onchange="applyFilters()">
                        <option value="all">All Platforms</option>
                        {platform_options}
                    </select>
                </div>
                <div class="filter-group">
                    <label>YOLO Version:</label>
                    <select id="filter-yolo-version" onchange="applyFilters()">
                        <option value="all">All Versions</option>
                        {version_options}
                    </select>
                </div>
                <div class="filter-group">
                    <label>Task:</label>
                    <select id="filter-task" onchange="applyFilters()">
                        <option value="all">All Tasks</option>
                        {task_options}
                    </select>
                </div>
                <div class="filter-group">
                    <label>LLM Size:</label>
                    <select id="filter-llm-size" onchange="applyFilters()">
                        <option value="all">All Sizes</option>
                        {size_options}
                    </select>
                </div>
                <div class="filter-group">
                    <label>Parameter Group:</label>
                    <select id="filter-param-group" onchange="applyFilters()">
                        <option value="all">All Groups</option>
                        {param_group_options}
                    </select>
                </div>
                <div class="filter-group">
                    <label>Architecture:</label>
                    <select id="filter-architecture" onchange="applyFilters()">
                        <option value="all">All (Dense + MoE)</option>
                        {arch_options}
                    </select>
                </div>
                <div class="filter-group">
                    <label>Specialization:</label>
                    <select id="filter-specialization" onchange="applyFilters()">
                        <option value="all">All (General + Code)</option>
                        {spec_options}
                    </select>
                </div>
                <button onclick="resetFilters()">Reset</button>
            </div>
        </section>"""

    def _generate_system_overview_section(self) -> str:
        """Generate system overview section."""
        if not self.platform_summaries:
            return '<section><h2>System Overview</h2><p class="no-data">No data available</p></section>'

        # Generate platform cards
        cards = ""
        for summary in self.platform_summaries:
            badge_class = {
                "jetson_nano": "badge-blue",
                "rpi_ai_hat_plus": "badge-green",
                "rpi_ai_hat_plus_2": "badge-red",
            }.get(summary.platform, "badge-blue")

            cards += f"""
            <div class="card">
                <h3><span class="badge {badge_class}">{summary.platform.replace('_', ' ').title()}</span></h3>
                <table>
                    <tr><td>CPU</td><td>{summary.cpu_model}</td></tr>
                    <tr><td>Accelerator</td><td>{summary.accelerator}</td></tr>
                    <tr><td>RAM</td><td>{summary.ram_size_gb} GB</td></tr>
                    <tr><td>OS</td><td>{summary.os_version}</td></tr>
                    <tr><td>Benchmark Runs</td><td>{summary.num_benchmark_runs}</td></tr>
                    <tr><td>YOLO Results</td><td>{summary.num_yolo_results}</td></tr>
                    <tr><td>LLM Results</td><td>{summary.num_llm_results}</td></tr>
                </table>
            </div>"""

        # Summary stats
        total_yolo = sum(s.num_yolo_results for s in self.platform_summaries)
        total_llm = sum(s.num_llm_results for s in self.platform_summaries)
        total_runs = sum(s.num_benchmark_runs for s in self.platform_summaries)

        return f"""
        <section id="system-overview">
            <h2>System Overview</h2>
            <div class="grid grid-4">
                <div class="stat-card card">
                    <div class="value">{len(self.platform_summaries)}</div>
                    <div class="label">Platforms</div>
                </div>
                <div class="stat-card card">
                    <div class="value">{total_runs}</div>
                    <div class="label">Benchmark Runs</div>
                </div>
                <div class="stat-card card">
                    <div class="value">{total_yolo}</div>
                    <div class="label">YOLO Results</div>
                </div>
                <div class="stat-card card">
                    <div class="value">{total_llm}</div>
                    <div class="label">LLM Results</div>
                </div>
            </div>
            <div class="grid grid-3" style="margin-top: 1rem;">
                {cards}
            </div>
        </section>"""

    def _generate_yolo_performance_section(self) -> str:
        """Generate YOLO performance charts section."""
        if not self.yolo_metrics:
            return '<section><h2>YOLO Performance</h2><p class="no-data">No YOLO data available</p></section>'

        return """
        <section id="yolo-performance">
            <h2>YOLO Performance</h2>
            <div class="grid grid-2">
                <div class="card">
                    <h3>Throughput (FPS) by Model</h3>
                    <div class="chart-container">
                        <canvas id="yolo-throughput-chart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>Latency (ms) by Model</h3>
                    <div class="chart-container">
                        <canvas id="yolo-latency-chart"></canvas>
                    </div>
                </div>
            </div>
            <div class="grid grid-2" style="margin-top: 1rem;">
                <div class="card">
                    <h3>Accuracy (mAP) by Model</h3>
                    <div class="chart-container">
                        <canvas id="yolo-accuracy-chart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>Power Consumption (W) by Model</h3>
                    <div class="chart-container">
                        <canvas id="yolo-power-chart"></canvas>
                    </div>
                </div>
            </div>
        </section>"""

    def _generate_yolo_scaling_section(self) -> str:
        """Generate YOLO scaling analysis section."""
        if not self.yolo_metrics:
            return ""

        return """
        <section id="yolo-scaling">
            <h2>YOLO Scaling Analysis</h2>
            <div class="grid grid-2">
                <div class="card">
                    <h3>Throughput vs Model Size</h3>
                    <div class="chart-container">
                        <canvas id="yolo-scaling-fps-chart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>Latency vs Accuracy Trade-off</h3>
                    <div class="chart-container">
                        <canvas id="yolo-tradeoff-chart"></canvas>
                    </div>
                </div>
            </div>
        </section>"""

    def _generate_llm_performance_section(self) -> str:
        """Generate LLM performance charts section."""
        if not self.llm_metrics:
            return '<section><h2>LLM Performance</h2><p class="no-data">No LLM data available</p></section>'

        return """
        <section id="llm-performance">
            <h2>LLM Performance</h2>
            <div class="grid grid-2">
                <div class="card">
                    <h3>Tokens per Second by Model</h3>
                    <div class="chart-container">
                        <canvas id="llm-tps-chart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>Time to First Token (ms) by Model</h3>
                    <div class="chart-container">
                        <canvas id="llm-ttft-chart"></canvas>
                    </div>
                </div>
            </div>
        </section>"""

    def _generate_llm_efficiency_section(self) -> str:
        """Generate LLM efficiency charts section."""
        if not self.llm_metrics:
            return ""

        return """
        <section id="llm-efficiency">
            <h2>LLM Efficiency</h2>
            <div class="grid grid-2">
                <div class="card">
                    <h3>Memory Usage by Model</h3>
                    <div class="chart-container">
                        <canvas id="llm-memory-chart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>Tokens/Second vs Memory Trade-off</h3>
                    <div class="chart-container">
                        <canvas id="llm-tradeoff-chart"></canvas>
                    </div>
                </div>
            </div>
        </section>"""

    def _generate_variance_section(self) -> str:
        """Generate stability/variance analysis section."""
        return """
        <section id="variance">
            <h2>Stability & Variance Analysis</h2>
            <div class="grid grid-2">
                <div class="card">
                    <h3>YOLO Latency Distribution</h3>
                    <div class="chart-container">
                        <canvas id="yolo-variance-chart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>LLM Tokens/Second Distribution</h3>
                    <div class="chart-container">
                        <canvas id="llm-variance-chart"></canvas>
                    </div>
                </div>
            </div>
        </section>"""

    def _generate_raw_data_section(self) -> str:
        """Generate raw data tables section.

        Model Expansion PRD - Phase 6 Task 6.2: Separate MoE and dense labeling.
        """
        # YOLO table rows
        yolo_rows = ""
        for m in self.yolo_metrics[:20]:  # Limit to 20 rows
            map_value = f"{m.map_mean:.4f}" if m.map_mean is not None else "N/A"
            yolo_rows += f"""
            <tr>
                <td>{m.model_name}</td>
                <td>{m.yolo_version}</td>
                <td>{m.task}</td>
                <td>{m.throughput_mean_fps:.2f}</td>
                <td>{m.latency_mean_ms:.2f}</td>
                <td>{map_value}</td>
                <td>{m.num_runs}</td>
            </tr>"""

        # LLM table rows with Phase 6 badges
        llm_rows = ""
        for m in self.llm_metrics[:20]:  # Limit to 20 rows
            memory_value = f"{m.memory_used_mb_mean:.1f}" if m.memory_used_mb_mean is not None else "N/A"

            # Phase 6: Add architecture and specialization badges
            badges = ""
            if m.architecture == "moe":
                badges += '<span class="badge badge-purple">MoE</span> '
            if m.specialization == "code":
                badges += '<span class="badge badge-orange">Code</span> '

            # Parameter group badge
            param_group = m.parameter_group or m.model_size
            param_group_class = {
                "1B": "badge-green",
                "3B": "badge-blue",
                "7B": "badge-blue",
                "8B": "badge-blue",
                "9B": "badge-red",
            }.get(param_group, "badge-blue")

            llm_rows += f"""
            <tr>
                <td>{m.model_name} {badges}</td>
                <td><span class="badge {param_group_class}">{param_group}</span></td>
                <td>{m.prompt_id}</td>
                <td>{m.tps_mean:.2f}</td>
                <td>{m.ttft_mean_ms:.2f}</td>
                <td>{m.ttft_median_ms:.2f}</td>
                <td>{memory_value}</td>
                <td>{m.num_runs}</td>
            </tr>"""

        return f"""
        <section id="raw-data">
            <h2>Raw Data</h2>
            <h3>YOLO Results</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Version</th>
                            <th>Task</th>
                            <th>FPS</th>
                            <th>Latency (ms)</th>
                            <th>mAP</th>
                            <th>Runs</th>
                        </tr>
                    </thead>
                    <tbody>
                        {yolo_rows if yolo_rows else '<tr><td colspan="7" class="no-data">No data</td></tr>'}
                    </tbody>
                </table>
            </div>

            <h3 style="margin-top: 1.5rem;">LLM Results</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Group</th>
                            <th>Prompt</th>
                            <th>TPS</th>
                            <th>TTFT Mean (ms)</th>
                            <th>TTFT Median (ms)</th>
                            <th>Memory (MB)</th>
                            <th>Runs</th>
                        </tr>
                    </thead>
                    <tbody>
                        {llm_rows if llm_rows else '<tr><td colspan="8" class="no-data">No data</td></tr>'}
                    </tbody>
                </table>
            </div>
        </section>"""

    def _generate_download_section(self) -> str:
        """Generate download links section."""
        return """
        <section id="downloads">
            <h2>Download Data</h2>
            <div class="download-links">
                <a href="#" onclick="downloadJSON()">Download JSON (Full Results)</a>
                <a href="#" onclick="downloadYOLOCSV()">Download YOLO CSV</a>
                <a href="#" onclick="downloadLLMCSV()">Download LLM CSV</a>
            </div>
        </section>"""

    def _generate_scripts(self) -> str:
        """Generate JavaScript for charts and interactivity."""
        # Prepare data for charts
        yolo_data = json.dumps([m.to_dict() for m in self.yolo_metrics])
        llm_data = json.dumps([m.to_dict() for m in self.llm_metrics])
        platform_colors = json.dumps(PLATFORM_COLORS)

        return f"""
    <script>
        // Data
        const yoloData = {yolo_data};
        const llmData = {llm_data};
        const platformColors = {platform_colors};

        // Chart instances
        let charts = {{}};

        // Initialize charts on load
        document.addEventListener('DOMContentLoaded', function() {{
            initializeCharts();
        }});

        function initializeCharts() {{
            // YOLO Throughput Chart
            if (yoloData.length > 0) {{
                const ctx1 = document.getElementById('yolo-throughput-chart');
                if (ctx1) {{
                    charts.yoloThroughput = new Chart(ctx1, {{
                        type: 'bar',
                        data: {{
                            labels: yoloData.map(d => d.model_name),
                            datasets: [{{
                                label: 'FPS',
                                data: yoloData.map(d => d.throughput_mean_fps),
                                backgroundColor: '#2563eb',
                                borderRadius: 4,
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'FPS' }} }}
                            }}
                        }}
                    }});
                }}

                // YOLO Latency Chart
                const ctx2 = document.getElementById('yolo-latency-chart');
                if (ctx2) {{
                    charts.yoloLatency = new Chart(ctx2, {{
                        type: 'bar',
                        data: {{
                            labels: yoloData.map(d => d.model_name),
                            datasets: [{{
                                label: 'Mean Latency',
                                data: yoloData.map(d => d.latency_mean_ms),
                                backgroundColor: '#16a34a',
                                borderRadius: 4,
                            }}, {{
                                label: 'P95 Latency',
                                data: yoloData.map(d => d.latency_p95_ms),
                                backgroundColor: '#ca8a04',
                                borderRadius: 4,
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'Latency (ms)' }} }}
                            }}
                        }}
                    }});
                }}

                // YOLO Accuracy Chart
                const ctx3 = document.getElementById('yolo-accuracy-chart');
                if (ctx3) {{
                    const accuracyData = yoloData.filter(d => d.map_mean !== null);
                    charts.yoloAccuracy = new Chart(ctx3, {{
                        type: 'bar',
                        data: {{
                            labels: accuracyData.map(d => d.model_name),
                            datasets: [{{
                                label: 'mAP',
                                data: accuracyData.map(d => d.map_mean),
                                backgroundColor: '#dc2626',
                                borderRadius: 4,
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, max: 1, title: {{ display: true, text: 'mAP' }} }}
                            }}
                        }}
                    }});
                }}

                // YOLO Power Chart
                const ctx4 = document.getElementById('yolo-power-chart');
                if (ctx4) {{
                    const powerData = yoloData.filter(d => d.power_watts_mean !== null);
                    charts.yoloPower = new Chart(ctx4, {{
                        type: 'bar',
                        data: {{
                            labels: powerData.map(d => d.model_name),
                            datasets: [{{
                                label: 'Power (W)',
                                data: powerData.map(d => d.power_watts_mean),
                                backgroundColor: '#7c3aed',
                                borderRadius: 4,
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'Power (W)' }} }}
                            }}
                        }}
                    }});
                }}

                // YOLO Scaling Chart
                const ctx5 = document.getElementById('yolo-scaling-fps-chart');
                if (ctx5) {{
                    // Group by model size
                    const sizeOrder = ['n', 's', 'm', 'l', 'x'];
                    const bySize = {{}};
                    yoloData.forEach(d => {{
                        const size = d.model_name.match(/([nsmlx])[\.-]/)?.[1] || 'n';
                        if (!bySize[size]) bySize[size] = [];
                        bySize[size].push(d.throughput_mean_fps);
                    }});

                    charts.yoloScaling = new Chart(ctx5, {{
                        type: 'line',
                        data: {{
                            labels: sizeOrder.filter(s => bySize[s]),
                            datasets: [{{
                                label: 'Average FPS',
                                data: sizeOrder.filter(s => bySize[s]).map(s =>
                                    bySize[s].reduce((a, b) => a + b, 0) / bySize[s].length
                                ),
                                borderColor: '#2563eb',
                                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                                fill: true,
                                tension: 0.3,
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'FPS' }} }},
                                x: {{ title: {{ display: true, text: 'Model Size' }} }}
                            }}
                        }}
                    }});
                }}

                // YOLO Trade-off Chart
                const ctx6 = document.getElementById('yolo-tradeoff-chart');
                if (ctx6) {{
                    const tradeoffData = yoloData.filter(d => d.map_mean !== null);
                    charts.yoloTradeoff = new Chart(ctx6, {{
                        type: 'scatter',
                        data: {{
                            datasets: [{{
                                label: 'Latency vs Accuracy',
                                data: tradeoffData.map(d => ({{ x: d.latency_mean_ms, y: d.map_mean }})),
                                backgroundColor: '#2563eb',
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{ title: {{ display: true, text: 'Latency (ms)' }} }},
                                y: {{ title: {{ display: true, text: 'mAP' }}, min: 0, max: 1 }}
                            }}
                        }}
                    }});
                }}

                // YOLO Variance Chart
                const ctx7 = document.getElementById('yolo-variance-chart');
                if (ctx7) {{
                    charts.yoloVariance = new Chart(ctx7, {{
                        type: 'bar',
                        data: {{
                            labels: yoloData.map(d => d.model_name),
                            datasets: [{{
                                label: 'Mean',
                                data: yoloData.map(d => d.latency_mean_ms),
                                backgroundColor: '#2563eb',
                            }}, {{
                                label: 'Std Dev',
                                data: yoloData.map(d => d.latency_std_ms),
                                backgroundColor: '#dc2626',
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'Latency (ms)' }} }}
                            }}
                        }}
                    }});
                }}
            }}

            // LLM Charts
            if (llmData.length > 0) {{
                // LLM TPS Chart
                const ctx8 = document.getElementById('llm-tps-chart');
                if (ctx8) {{
                    charts.llmTps = new Chart(ctx8, {{
                        type: 'bar',
                        data: {{
                            labels: llmData.map(d => d.model_name + ' (' + d.prompt_id + ')'),
                            datasets: [{{
                                label: 'Tokens/Second',
                                data: llmData.map(d => d.tps_mean),
                                backgroundColor: '#16a34a',
                                borderRadius: 4,
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'Tokens/Second' }} }}
                            }}
                        }}
                    }});
                }}

                // LLM TTFT Chart
                const ctx9 = document.getElementById('llm-ttft-chart');
                if (ctx9) {{
                    charts.llmTtft = new Chart(ctx9, {{
                        type: 'bar',
                        data: {{
                            labels: llmData.map(d => d.model_name + ' (' + d.prompt_id + ')'),
                            datasets: [{{
                                label: 'TTFT (ms)',
                                data: llmData.map(d => d.ttft_mean_ms),
                                backgroundColor: '#ca8a04',
                                borderRadius: 4,
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'TTFT (ms)' }} }}
                            }}
                        }}
                    }});
                }}

                // LLM Memory Chart
                const ctx10 = document.getElementById('llm-memory-chart');
                if (ctx10) {{
                    const memData = llmData.filter(d => d.memory_used_mb_mean !== null);
                    charts.llmMemory = new Chart(ctx10, {{
                        type: 'bar',
                        data: {{
                            labels: memData.map(d => d.model_name),
                            datasets: [{{
                                label: 'Memory (MB)',
                                data: memData.map(d => d.memory_used_mb_mean),
                                backgroundColor: '#7c3aed',
                                borderRadius: 4,
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'Memory (MB)' }} }}
                            }}
                        }}
                    }});
                }}

                // LLM Trade-off Chart
                const ctx11 = document.getElementById('llm-tradeoff-chart');
                if (ctx11) {{
                    const tradeoffData = llmData.filter(d => d.memory_used_mb_mean !== null);
                    charts.llmTradeoff = new Chart(ctx11, {{
                        type: 'scatter',
                        data: {{
                            datasets: [{{
                                label: 'TPS vs Memory',
                                data: tradeoffData.map(d => ({{ x: d.memory_used_mb_mean, y: d.tps_mean }})),
                                backgroundColor: '#16a34a',
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{ title: {{ display: true, text: 'Memory (MB)' }} }},
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'Tokens/Second' }} }}
                            }}
                        }}
                    }});
                }}

                // LLM Variance Chart
                const ctx12 = document.getElementById('llm-variance-chart');
                if (ctx12) {{
                    charts.llmVariance = new Chart(ctx12, {{
                        type: 'bar',
                        data: {{
                            labels: llmData.map(d => d.model_name + ' (' + d.prompt_id + ')'),
                            datasets: [{{
                                label: 'Mean TPS',
                                data: llmData.map(d => d.tps_mean),
                                backgroundColor: '#16a34a',
                            }}, {{
                                label: 'Std Dev',
                                data: llmData.map(d => d.tps_std),
                                backgroundColor: '#dc2626',
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'Tokens/Second' }} }}
                            }}
                        }}
                    }});
                }}
            }}
        }}

        function applyFilters() {{
            // Get filter values
            const platform = document.getElementById('filter-platform').value;
            const yoloVersion = document.getElementById('filter-yolo-version').value;
            const task = document.getElementById('filter-task').value;
            const llmSize = document.getElementById('filter-llm-size').value;
            // Phase 6: New filter values
            const paramGroup = document.getElementById('filter-param-group')?.value || 'all';
            const architecture = document.getElementById('filter-architecture')?.value || 'all';
            const specialization = document.getElementById('filter-specialization')?.value || 'all';

            // Filter YOLO data
            let filteredYolo = yoloData;
            if (yoloVersion !== 'all') {{
                filteredYolo = filteredYolo.filter(d => d.yolo_version === yoloVersion);
            }}
            if (task !== 'all') {{
                filteredYolo = filteredYolo.filter(d => d.task === task);
            }}

            // Filter LLM data
            let filteredLlm = llmData;
            if (llmSize !== 'all') {{
                filteredLlm = filteredLlm.filter(d => d.model_size === llmSize);
            }}
            // Phase 6: Apply new filters
            if (paramGroup !== 'all') {{
                filteredLlm = filteredLlm.filter(d => (d.parameter_group || d.model_size) === paramGroup);
            }}
            if (architecture !== 'all') {{
                filteredLlm = filteredLlm.filter(d => d.architecture === architecture);
            }}
            if (specialization !== 'all') {{
                filteredLlm = filteredLlm.filter(d => d.specialization === specialization);
            }}

            // Update charts with filtered data
            updateChartsWithData(filteredYolo, filteredLlm);
        }}

        function resetFilters() {{
            document.getElementById('filter-platform').value = 'all';
            document.getElementById('filter-yolo-version').value = 'all';
            document.getElementById('filter-task').value = 'all';
            document.getElementById('filter-llm-size').value = 'all';
            // Phase 6: Reset new filters
            if (document.getElementById('filter-param-group')) {{
                document.getElementById('filter-param-group').value = 'all';
            }}
            if (document.getElementById('filter-architecture')) {{
                document.getElementById('filter-architecture').value = 'all';
            }}
            if (document.getElementById('filter-specialization')) {{
                document.getElementById('filter-specialization').value = 'all';
            }}
            updateChartsWithData(yoloData, llmData);
        }}

        function updateChartsWithData(yolo, llm) {{
            // Update YOLO charts
            if (charts.yoloThroughput) {{
                charts.yoloThroughput.data.labels = yolo.map(d => d.model_name);
                charts.yoloThroughput.data.datasets[0].data = yolo.map(d => d.throughput_mean_fps);
                charts.yoloThroughput.update();
            }}
            if (charts.yoloLatency) {{
                charts.yoloLatency.data.labels = yolo.map(d => d.model_name);
                charts.yoloLatency.data.datasets[0].data = yolo.map(d => d.latency_mean_ms);
                charts.yoloLatency.data.datasets[1].data = yolo.map(d => d.latency_p95_ms);
                charts.yoloLatency.update();
            }}

            // Update LLM charts
            if (charts.llmTps) {{
                charts.llmTps.data.labels = llm.map(d => d.model_name + ' (' + d.prompt_id + ')');
                charts.llmTps.data.datasets[0].data = llm.map(d => d.tps_mean);
                charts.llmTps.update();
            }}
            if (charts.llmTtft) {{
                charts.llmTtft.data.labels = llm.map(d => d.model_name + ' (' + d.prompt_id + ')');
                charts.llmTtft.data.datasets[0].data = llm.map(d => d.ttft_mean_ms);
                charts.llmTtft.update();
            }}
        }}

        function downloadJSON() {{
            const data = {{
                yolo: yoloData,
                llm: llmData,
                generated: new Date().toISOString()
            }};
            downloadFile(JSON.stringify(data, null, 2), 'benchmark_results.json', 'application/json');
        }}

        function downloadYOLOCSV() {{
            if (yoloData.length === 0) return;
            const headers = Object.keys(yoloData[0]);
            const csv = [
                headers.join(','),
                ...yoloData.map(row => headers.map(h => row[h] ?? '').join(','))
            ].join('\\n');
            downloadFile(csv, 'yolo_results.csv', 'text/csv');
        }}

        function downloadLLMCSV() {{
            if (llmData.length === 0) return;
            const headers = Object.keys(llmData[0]);
            const csv = [
                headers.join(','),
                ...llmData.map(row => headers.map(h => row[h] ?? '').join(','))
            ].join('\\n');
            downloadFile(csv, 'llm_results.csv', 'text/csv');
        }}

        function downloadFile(content, filename, mimeType) {{
            const blob = new Blob([content], {{ type: mimeType }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
    </script>"""


def generate_dashboard(
    input_dir: Path,
    output_path: Path,
    title: str = "Edge AI Benchmark Dashboard",
) -> None:
    """Convenience function to generate dashboard from results directory.

    Args:
        input_dir: Directory containing raw result JSON files
        output_path: Path to write the HTML dashboard
        title: Dashboard title
    """
    aggregator = ResultsAggregator()
    aggregator.load_directory(input_dir)

    generator = DashboardGenerator(aggregator)
    generator.generate(output_path, title)
