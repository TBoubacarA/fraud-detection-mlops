"""
Module de génération de dashboards HTML pour le monitoring
Génère des tableaux de bord interactifs pour visualiser les métriques
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """Générateur de dashboards HTML"""
    
    def __init__(self, metrics_collector, alert_manager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
    
    def generate_main_dashboard(self) -> str:
        """Génère le dashboard principal"""
        try:
            # Récupérer les métriques actuelles
            metrics = self.metrics_collector.get_metrics()
            endpoint_stats = self.metrics_collector.get_endpoint_stats()
            hourly_stats = self.metrics_collector.get_hourly_stats(24)
            alerts = self.alert_manager.get_alerts(10)
            
            # Générer le HTML
            html_content = self._generate_html_template(
                title="Dashboard de Monitoring - API Fraud Detection",
                content=self._generate_main_content(metrics, endpoint_stats, hourly_stats, alerts)
            )
            
            return html_content
            
        except Exception as e:
            logger.error(f"Erreur génération dashboard: {e}")
            return self._generate_error_page(str(e))
    
    def generate_security_dashboard(self) -> str:
        """Génère le dashboard de sécurité"""
        try:
            metrics = self.metrics_collector.get_metrics()
            alerts = self.alert_manager.get_alerts(50)
            
            # Filtrer les alertes de sécurité
            security_alerts = [a for a in alerts if 'security' in a.get('name', '').lower()]
            
            content = self._generate_security_content(metrics, security_alerts)
            
            return self._generate_html_template(
                title="Dashboard de Sécurité - API Fraud Detection",
                content=content
            )
            
        except Exception as e:
            logger.error(f"Erreur génération dashboard sécurité: {e}")
            return self._generate_error_page(str(e))
    
    def generate_performance_dashboard(self) -> str:
        """Génère le dashboard de performance"""
        try:
            metrics = self.metrics_collector.get_metrics()
            endpoint_stats = self.metrics_collector.get_endpoint_stats()
            hourly_stats = self.metrics_collector.get_hourly_stats(24)
            
            content = self._generate_performance_content(metrics, endpoint_stats, hourly_stats)
            
            return self._generate_html_template(
                title="Dashboard de Performance - API Fraud Detection",
                content=content
            )
            
        except Exception as e:
            logger.error(f"Erreur génération dashboard performance: {e}")
            return self._generate_error_page(str(e))
    
    def _generate_html_template(self, title: str, content: str) -> str:
        """Génère le template HTML de base"""
        return f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                {self._get_css_styles()}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>{title}</h1>
                    <div class="timestamp">Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </header>
                
                <nav>
                    <a href="#" onclick="refreshData()" class="refresh-btn">🔄 Actualiser</a>
                    <a href="/dashboard" class="nav-link">Dashboard Principal</a>
                    <a href="/dashboard/security" class="nav-link">Sécurité</a>
                    <a href="/dashboard/performance" class="nav-link">Performance</a>
                </nav>
                
                <main>
                    {content}
                </main>
                
                <footer>
                    <p>API Fraud Detection - Dashboard de Monitoring</p>
                </footer>
            </div>
            
            <script>
                {self._get_javascript_code()}
            </script>
        </body>
        </html>
        """
    
    def _generate_main_content(self, metrics: Dict, endpoint_stats: Dict, hourly_stats: Dict, alerts: List) -> str:
        """Génère le contenu du dashboard principal"""
        api_metrics = metrics.get('api', {})
        model_metrics = metrics.get('model', {})
        system_metrics = metrics.get('system', {})
        security_metrics = metrics.get('security', {})
        
        # Calculer les statistiques
        total_requests = api_metrics.get('total_requests', 0)
        error_rate = 0
        if total_requests > 0:
            error_rate = (api_metrics.get('failed_requests', 0) / total_requests) * 100
        
        fraud_rate = model_metrics.get('fraud_rate', 0) * 100
        
        return f"""
        <div class="dashboard-grid">
            <!-- Métriques principales -->
            <div class="metrics-row">
                <div class="metric-card">
                    <h3>Requêtes Totales</h3>
                    <div class="metric-value">{total_requests:,}</div>
                    <div class="metric-label">Depuis le démarrage</div>
                </div>
                
                <div class="metric-card {'alert' if error_rate > 5 else ''}">
                    <h3>Taux d'Erreur</h3>
                    <div class="metric-value">{error_rate:.1f}%</div>
                    <div class="metric-label">{api_metrics.get('failed_requests', 0):,} erreurs</div>
                </div>
                
                <div class="metric-card">
                    <h3>Temps de Réponse</h3>
                    <div class="metric-value">{api_metrics.get('avg_response_time', 0):.3f}s</div>
                    <div class="metric-label">P95: {api_metrics.get('p95_response_time', 0):.3f}s</div>
                </div>
                
                <div class="metric-card">
                    <h3>Prédictions</h3>
                    <div class="metric-value">{model_metrics.get('total_predictions', 0):,}</div>
                    <div class="metric-label">Fraude: {fraud_rate:.1f}%</div>
                </div>
            </div>
            
            <!-- Métriques système -->
            <div class="system-metrics">
                <h2>Métriques Système</h2>
                <div class="metrics-row">
                    <div class="metric-card {'alert' if system_metrics.get('cpu_usage', 0) > 80 else ''}">
                        <h3>CPU</h3>
                        <div class="metric-value">{system_metrics.get('cpu_usage', 0):.1f}%</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {system_metrics.get('cpu_usage', 0)}%"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card {'alert' if system_metrics.get('memory_usage', 0) > 85 else ''}">
                        <h3>Mémoire</h3>
                        <div class="metric-value">{system_metrics.get('memory_usage', 0):.1f}%</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {system_metrics.get('memory_usage', 0)}%"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Connexions</h3>
                        <div class="metric-value">{system_metrics.get('active_connections', 0)}</div>
                        <div class="metric-label">Actives</div>
                    </div>
                </div>
            </div>
            
            <!-- Graphiques -->
            <div class="charts-section">
                <div class="chart-container">
                    <h2>Évolution des Requêtes</h2>
                    <div id="requests-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>Statistiques par Endpoint</h2>
                    <div id="endpoints-chart"></div>
                </div>
            </div>
            
            <!-- Alertes récentes -->
            <div class="alerts-section">
                <h2>Alertes Récentes</h2>
                {self._generate_alerts_table(alerts)}
            </div>
            
            <!-- Détails des endpoints -->
            <div class="endpoints-section">
                <h2>Détails des Endpoints</h2>
                {self._generate_endpoints_table(endpoint_stats)}
            </div>
        </div>
        
        <script>
            // Données pour les graphiques
            const hourlyData = {json.dumps(hourly_stats)};
            const endpointData = {json.dumps(endpoint_stats)};
            
            // Générer les graphiques
            generateRequestsChart(hourlyData);
            generateEndpointsChart(endpointData);
        </script>
        """
    
    def _generate_security_content(self, metrics: Dict, security_alerts: List) -> str:
        """Génère le contenu du dashboard de sécurité"""
        security_metrics = metrics.get('security', {})
        
        return f"""
        <div class="security-dashboard">
            <div class="security-metrics">
                <h2>Métriques de Sécurité</h2>
                <div class="metrics-row">
                    <div class="metric-card {'alert' if security_metrics.get('failed_authentications', 0) > 10 else ''}">
                        <h3>Authentifications Échouées</h3>
                        <div class="metric-value">{security_metrics.get('failed_authentications', 0)}</div>
                        <div class="metric-label">Dernières 24h</div>
                    </div>
                    
                    <div class="metric-card {'alert' if security_metrics.get('rate_limited_requests', 0) > 50 else ''}">
                        <h3>Requêtes Limitées</h3>
                        <div class="metric-value">{security_metrics.get('rate_limited_requests', 0)}</div>
                        <div class="metric-label">Rate limiting</div>
                    </div>
                    
                    <div class="metric-card {'alert' if security_metrics.get('malicious_requests', 0) > 0 else ''}">
                        <h3>Requêtes Malicieuses</h3>
                        <div class="metric-value">{security_metrics.get('malicious_requests', 0)}</div>
                        <div class="metric-label">Détectées</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>IPs Bloquées</h3>
                        <div class="metric-value">{security_metrics.get('blocked_ips', 0)}</div>
                        <div class="metric-label">Actives</div>
                    </div>
                </div>
            </div>
            
            <div class="security-alerts">
                <h2>Alertes de Sécurité</h2>
                {self._generate_security_alerts_table(security_alerts)}
            </div>
            
            <div class="security-chart">
                <h2>Évolution des Incidents de Sécurité</h2>
                <div id="security-chart"></div>
            </div>
        </div>
        """
    
    def _generate_performance_content(self, metrics: Dict, endpoint_stats: Dict, hourly_stats: Dict) -> str:
        """Génère le contenu du dashboard de performance"""
        api_metrics = metrics.get('api', {})
        model_metrics = metrics.get('model', {})
        
        return f"""
        <div class="performance-dashboard">
            <div class="performance-metrics">
                <h2>Métriques de Performance</h2>
                <div class="metrics-row">
                    <div class="metric-card">
                        <h3>Temps de Réponse Moyen</h3>
                        <div class="metric-value">{api_metrics.get('avg_response_time', 0):.3f}s</div>
                        <div class="metric-label">Toutes requêtes</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>P95 Temps de Réponse</h3>
                        <div class="metric-value">{api_metrics.get('p95_response_time', 0):.3f}s</div>
                        <div class="metric-label">95e percentile</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Requêtes/Seconde</h3>
                        <div class="metric-value">{api_metrics.get('requests_per_second', 0):.1f}</div>
                        <div class="metric-label">Débit actuel</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Temps de Prédiction</h3>
                        <div class="metric-value">{model_metrics.get('avg_prediction_time', 0):.3f}s</div>
                        <div class="metric-label">Modèle ML</div>
                    </div>
                </div>
            </div>
            
            <div class="performance-charts">
                <div class="chart-container">
                    <h2>Temps de Réponse par Endpoint</h2>
                    <div id="response-times-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h2>Débit des Requêtes</h2>
                    <div id="throughput-chart"></div>
                </div>
            </div>
            
            <div class="endpoint-performance">
                <h2>Performance par Endpoint</h2>
                {self._generate_performance_table(endpoint_stats)}
            </div>
        </div>
        """
    
    def _generate_alerts_table(self, alerts: List) -> str:
        """Génère le tableau des alertes"""
        if not alerts:
            return "<p>Aucune alerte récente</p>"
        
        table_rows = ""
        for alert in alerts[-10:]:  # Dernières 10 alertes
            severity_class = f"severity-{alert.get('severity', 'info')}"
            timestamp = alert.get('timestamp', '')
            
            table_rows += f"""
            <tr class="{severity_class}">
                <td>{timestamp}</td>
                <td>{alert.get('name', '')}</td>
                <td>{alert.get('message', '')}</td>
                <td><span class="severity {severity_class}">{alert.get('severity', '').upper()}</span></td>
            </tr>
            """
        
        return f"""
        <table class="alerts-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Nom</th>
                    <th>Message</th>
                    <th>Sévérité</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
    
    def _generate_endpoints_table(self, endpoint_stats: Dict) -> str:
        """Génère le tableau des endpoints"""
        if not endpoint_stats:
            return "<p>Aucune statistique d'endpoint disponible</p>"
        
        table_rows = ""
        for endpoint, stats in endpoint_stats.items():
            error_rate = stats.get('error_rate', 0) * 100
            error_class = "error-high" if error_rate > 5 else ""
            
            table_rows += f"""
            <tr class="{error_class}">
                <td>{endpoint}</td>
                <td>{stats.get('requests', 0):,}</td>
                <td>{stats.get('errors', 0):,}</td>
                <td>{error_rate:.1f}%</td>
                <td>{stats.get('avg_response_time', 0):.3f}s</td>
                <td>{stats.get('p95_response_time', 0):.3f}s</td>
            </tr>
            """
        
        return f"""
        <table class="endpoints-table">
            <thead>
                <tr>
                    <th>Endpoint</th>
                    <th>Requêtes</th>
                    <th>Erreurs</th>
                    <th>Taux d'Erreur</th>
                    <th>Temps Moyen</th>
                    <th>P95</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
    
    def _generate_security_alerts_table(self, security_alerts: List) -> str:
        """Génère le tableau des alertes de sécurité"""
        if not security_alerts:
            return "<p>Aucune alerte de sécurité récente</p>"
        
        table_rows = ""
        for alert in security_alerts[-20:]:  # Dernières 20 alertes
            severity_class = f"severity-{alert.get('severity', 'info')}"
            timestamp = alert.get('timestamp', '')
            
            table_rows += f"""
            <tr class="{severity_class}">
                <td>{timestamp}</td>
                <td>{alert.get('name', '')}</td>
                <td>{alert.get('message', '')}</td>
                <td><span class="severity {severity_class}">{alert.get('severity', '').upper()}</span></td>
            </tr>
            """
        
        return f"""
        <table class="security-alerts-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Type</th>
                    <th>Description</th>
                    <th>Sévérité</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
    
    def _generate_performance_table(self, endpoint_stats: Dict) -> str:
        """Génère le tableau de performance"""
        if not endpoint_stats:
            return "<p>Aucune statistique de performance disponible</p>"
        
        table_rows = ""
        for endpoint, stats in endpoint_stats.items():
            avg_time = stats.get('avg_response_time', 0)
            p95_time = stats.get('p95_response_time', 0)
            
            slow_class = "slow-endpoint" if avg_time > 0.5 else ""
            
            table_rows += f"""
            <tr class="{slow_class}">
                <td>{endpoint}</td>
                <td>{stats.get('requests', 0):,}</td>
                <td>{avg_time:.3f}s</td>
                <td>{p95_time:.3f}s</td>
                <td>{stats.get('requests', 0) / max(avg_time, 0.001):.1f}</td>
            </tr>
            """
        
        return f"""
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Endpoint</th>
                    <th>Requêtes</th>
                    <th>Temps Moyen</th>
                    <th>P95</th>
                    <th>Req/s</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
    
    def _get_css_styles(self) -> str:
        """Retourne les styles CSS pour le dashboard"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .timestamp {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        nav {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .nav-link, .refresh-btn {
            display: inline-block;
            padding: 10px 20px;
            margin-right: 10px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background 0.3s;
        }
        
        .nav-link:hover, .refresh-btn:hover {
            background: #5a6fd8;
        }
        
        .dashboard-grid {
            display: grid;
            gap: 20px;
        }
        
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-card.alert {
            border-left: 5px solid #ff6b6b;
            background: #fff5f5;
        }
        
        .metric-card h3 {
            color: #666;
            font-size: 1em;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #888;
            font-size: 0.9em;
        }
        
        .progress-bar {
            width: 100%;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s;
        }
        
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .chart-container h2 {
            margin-bottom: 15px;
            color: #333;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th {
            background: #f8f9fa;
            font-weight: bold;
            color: #555;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .severity-critical {
            background: #ff6b6b !important;
            color: white;
        }
        
        .severity-warning {
            background: #ffa726 !important;
            color: white;
        }
        
        .severity-info {
            background: #42a5f5 !important;
            color: white;
        }
        
        .severity {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .error-high {
            background: #fff5f5;
            border-left: 3px solid #ff6b6b;
        }
        
        .slow-endpoint {
            background: #fff8e1;
            border-left: 3px solid #ffa726;
        }
        
        footer {
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #666;
            background: white;
            border-radius: 10px;
        }
        
        @media (max-width: 768px) {
            .metrics-row {
                grid-template-columns: 1fr;
            }
            
            .container {
                padding: 10px;
            }
            
            header h1 {
                font-size: 2em;
            }
        }
        """
    
    def _get_javascript_code(self) -> str:
        """Retourne le code JavaScript pour les graphiques"""
        return """
        function refreshData() {
            location.reload();
        }
        
        function generateRequestsChart(hourlyData) {
            const hours = Object.keys(hourlyData).sort();
            const requests = hours.map(hour => hourlyData[hour].requests || 0);
            
            const data = [{
                x: hours,
                y: requests,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Requêtes',
                line: {color: '#667eea', width: 3}
            }];
            
            const layout = {
                title: '',
                xaxis: {title: 'Heure'},
                yaxis: {title: 'Nombre de requêtes'},
                margin: {l: 50, r: 50, t: 20, b: 50}
            };
            
            Plotly.newPlot('requests-chart', data, layout, {responsive: true});
        }
        
        function generateEndpointsChart(endpointData) {
            const endpoints = Object.keys(endpointData);
            const requests = endpoints.map(ep => endpointData[ep].requests || 0);
            
            const data = [{
                x: endpoints,
                y: requests,
                type: 'bar',
                marker: {color: '#667eea'}
            }];
            
            const layout = {
                title: '',
                xaxis: {title: 'Endpoint'},
                yaxis: {title: 'Requêtes'},
                margin: {l: 50, r: 50, t: 20, b: 100}
            };
            
            Plotly.newPlot('endpoints-chart', data, layout, {responsive: true});
        }
        
        // Auto-refresh toutes les 30 secondes
        setInterval(refreshData, 30000);
        """
    
    def _generate_error_page(self, error_message: str) -> str:
        """Génère une page d'erreur"""
        return f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Erreur - Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                    padding: 50px;
                    text-align: center;
                }}
                .error-container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    max-width: 500px;
                    margin: 0 auto;
                }}
                h1 {{
                    color: #ff6b6b;
                    margin-bottom: 20px;
                }}
                p {{
                    color: #666;
                    line-height: 1.6;
                }}
                .retry-btn {{
                    background: #667eea;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>Erreur de Dashboard</h1>
                <p>Une erreur s'est produite lors de la génération du dashboard:</p>
                <p><strong>{error_message}</strong></p>
                <button class="retry-btn" onclick="location.reload()">Réessayer</button>
            </div>
        </body>
        </html>
        """
    
    def save_dashboard(self, content: str, filename: str) -> bool:
        """Sauvegarde le dashboard dans un fichier"""
        try:
            dashboards_dir = Path("dashboards")
            dashboards_dir.mkdir(exist_ok=True)
            
            file_path = dashboards_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Dashboard sauvegardé: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde dashboard: {e}")
            return False