from flask import Flask, render_template, jsonify
import json
from fpga_simulator import AdvancedFPGAFabric

app = Flask(__name__)
fpga = AdvancedFPGAFabric(rows=8, cols=8)

@app.route('/')
def index():
    return render_template('fpga_web_interface.html')

@app.route('/api/metrics')
def get_metrics():
    metrics = fpga.get_performance_metrics()
    return jsonify(metrics)

@app.route('/api/inject_fault', methods=['POST'])
def inject_fault():
    stats = fpga.inject_and_recover_faults(num_faults=1)
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)