"""
Admin Dashboard for Telecom Churn Prediction System
Provides web interface for monitoring and alerts
"""

from flask import Flask, render_template, jsonify, request
import json
from datetime import datetime
import config


class AdminDashboard:
    """Admin dashboard for monitoring and alerts"""
    
    def __init__(self, psi_monitor, ensemble_model=None):
        """
        Initialize admin dashboard
        
        Args:
            psi_monitor: PSIMonitor or BackgroundPSIMonitor instance
            ensemble_model: Trained ensemble model (optional)
        """
        self.app = Flask(__name__)
        self.psi_monitor = psi_monitor
        self.ensemble_model = ensemble_model
        self.prediction_history = []
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/')
        def index():
            """Dashboard home page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get current alerts"""
            alerts = self.psi_monitor.get_alerts()
            return jsonify({
                'alerts': alerts,
                'count': len(alerts)
            })
        
        @self.app.route('/api/alerts/<severity>')
        def get_alerts_by_severity(severity):
            """Get alerts by severity"""
            alerts = self.psi_monitor.get_alerts(severity=severity)
            return jsonify({
                'severity': severity,
                'alerts': alerts,
                'count': len(alerts)
            })
        
        @self.app.route('/api/drift/status')
        def get_drift_status():
            """Get current drift status"""
            if hasattr(self.psi_monitor, 'drift_history'):
                # BackgroundPSIMonitor
                history = self.psi_monitor.get_drift_history(limit=1)
                if history:
                    return jsonify(history[0])
                else:
                    return jsonify({'status': 'no_data', 'message': 'No drift checks performed yet'})
            else:
                return jsonify({'status': 'manual_mode', 'message': 'Use /api/drift/check endpoint'})
        
        @self.app.route('/api/drift/history')
        def get_drift_history():
            """Get drift check history"""
            limit = request.args.get('limit', type=int, default=10)
            
            if hasattr(self.psi_monitor, 'drift_history'):
                history = self.psi_monitor.get_drift_history(limit=limit)
                return jsonify({
                    'history': history,
                    'count': len(history)
                })
            else:
                return jsonify({'error': 'Drift history not available in manual mode'})
        
        @self.app.route('/api/predictions/history')
        def get_prediction_history():
            """Get prediction history"""
            limit = request.args.get('limit', type=int, default=20)
            return jsonify({
                'predictions': self.prediction_history[-limit:],
                'count': len(self.prediction_history)
            })
        
        @self.app.route('/api/predict', methods=['POST'])
        def predict():
            """Make a prediction"""
            if self.ensemble_model is None:
                return jsonify({'error': 'Model not loaded'}), 400
            
            try:
                data = request.json
                features = data.get('features')
                
                if features is None:
                    return jsonify({'error': 'No features provided'}), 400
                
                # Make prediction
                prediction = self.ensemble_model.predict([features])[0]
                proba = self.ensemble_model.predict_proba([features])[0]
                
                result = {
                    'prediction': int(prediction),
                    'probability': {
                        'no_churn': float(proba[0]),
                        'churn': float(proba[1])
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store in history
                self.prediction_history.append(result)
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/info')
        def system_info():
            """Get system information"""
            info = {
                'ensemble_weights': config.ENSEMBLE_WEIGHTS,
                'psi_thresholds': {
                    'warning': config.PSI_THRESHOLD_WARNING,
                    'critical': config.PSI_THRESHOLD_CRITICAL
                },
                'model_loaded': self.ensemble_model is not None,
                'total_predictions': len(self.prediction_history),
                'total_alerts': len(self.psi_monitor.get_alerts())
            }
            return jsonify(info)
    
    def run(self, host=None, port=None, debug=None):
        """
        Start the dashboard server
        
        Args:
            host: Server host
            port: Server port
            debug: Debug mode
        """
        host = host or config.DASHBOARD_HOST
        port = port or config.DASHBOARD_PORT
        debug = debug if debug is not None else config.DASHBOARD_DEBUG
        
        print(f"Starting Admin Dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_dashboard_template():
    """Create HTML template for dashboard"""
    template_dir = 'templates'
    import os
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telecom Churn Prediction - Admin Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; margin-bottom: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h2 { color: #555; margin-bottom: 15px; font-size: 1.3em; }
        .alert { padding: 12px; margin: 10px 0; border-radius: 4px; }
        .alert-warning { background: #fff3cd; border-left: 4px solid #ffc107; }
        .alert-critical { background: #f8d7da; border-left: 4px solid #dc3545; }
        .alert-stable { background: #d4edda; border-left: 4px solid #28a745; }
        .status-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: bold; }
        .status-stable { background: #28a745; color: white; }
        .status-warning { background: #ffc107; color: #333; }
        .status-critical { background: #dc3545; color: white; }
        .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .info-item { padding: 15px; background: #f8f9fa; border-radius: 4px; }
        .info-label { font-size: 0.9em; color: #666; }
        .info-value { font-size: 1.5em; font-weight: bold; color: #333; margin-top: 5px; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .refresh-time { color: #999; font-size: 0.9em; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔔 Telecom Churn Prediction - Admin Dashboard</h1>
        
        <div class="card">
            <h2>System Status</h2>
            <div class="info-grid" id="systemInfo">
                <div class="info-item">
                    <div class="info-label">Model Status</div>
                    <div class="info-value" id="modelStatus">Loading...</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Total Predictions</div>
                    <div class="info-value" id="totalPredictions">-</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Active Alerts</div>
                    <div class="info-value" id="totalAlerts">-</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Current Drift Status</h2>
            <div id="driftStatus">
                <p>Loading drift status...</p>
            </div>
        </div>
        
        <div class="card">
            <h2>Recent Alerts</h2>
            <button onclick="refreshAlerts()">Refresh Alerts</button>
            <div id="alertsList" style="margin-top: 15px;">
                <p>Loading alerts...</p>
            </div>
            <div class="refresh-time" id="alertsRefreshTime"></div>
        </div>
    </div>
    
    <script>
        function updateSystemInfo() {
            fetch('/api/system/info')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('modelStatus').textContent = data.model_loaded ? '✓ Loaded' : '✗ Not Loaded';
                    document.getElementById('totalPredictions').textContent = data.total_predictions;
                    document.getElementById('totalAlerts').textContent = data.total_alerts;
                });
        }
        
        function updateDriftStatus() {
            fetch('/api/drift/status')
                .then(res => res.json())
                .then(data => {
                    const container = document.getElementById('driftStatus');
                    if (data.overall_status) {
                        let statusClass = 'status-' + data.overall_status;
                        let html = `<div><span class="status-badge ${statusClass}">${data.overall_status.toUpperCase()}</span></div>`;
                        
                        if (data.features) {
                            html += '<div style="margin-top: 15px;">';
                            for (const [feature, info] of Object.entries(data.features)) {
                                html += `<div style="margin: 5px 0;"><strong>${feature}:</strong> PSI = ${info.psi.toFixed(4)} (${info.status})</div>`;
                            }
                            html += '</div>';
                        }
                        container.innerHTML = html;
                    } else {
                        container.innerHTML = `<p>${data.message || 'No drift data available'}</p>`;
                    }
                });
        }
        
        function refreshAlerts() {
            fetch('/api/alerts')
                .then(res => res.json())
                .then(data => {
                    const container = document.getElementById('alertsList');
                    if (data.alerts.length === 0) {
                        container.innerHTML = '<div class="alert alert-stable">No active alerts - system is stable</div>';
                    } else {
                        let html = '';
                        data.alerts.forEach(alert => {
                            let alertClass = alert.severity === 'critical' ? 'alert-critical' : 'alert-warning';
                            html += `<div class="alert ${alertClass}">
                                <strong>${alert.severity.toUpperCase()}:</strong> ${alert.feature}<br>
                                PSI: ${alert.psi.toFixed(4)} | Time: ${new Date(alert.timestamp).toLocaleString()}
                            </div>`;
                        });
                        container.innerHTML = html;
                    }
                    document.getElementById('alertsRefreshTime').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
                });
        }
        
        // Initial load
        updateSystemInfo();
        updateDriftStatus();
        refreshAlerts();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            updateSystemInfo();
            updateDriftStatus();
            refreshAlerts();
        }, 30000);
    </script>
</body>
</html>"""
    
    with open(f'{template_dir}/dashboard.html', 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard template created at {template_dir}/dashboard.html")
