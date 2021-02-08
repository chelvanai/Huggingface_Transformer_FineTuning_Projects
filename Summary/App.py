from flask import Flask, request, Response, jsonify
from test import Summary

app = Flask(__name__)

summary_obj = Summary()


@app.route('/summary', methods=['GET', 'POST'])
def summary():
    data = request.form['ctext']
    summary_text = summary_obj.getSummary(str(data))

    final_result = {"Original Text": str(data), "Summary Text": str(summary_text)}
    return jsonify(final_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
