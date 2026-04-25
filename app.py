from flask import Flask, render_template, request
import pandas as pd
from model import train_model
from fairness import check_bias
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        target = request.form["target"]
        sensitive = request.form["sensitive"]

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        model, X_test, y_test, preds = train_model(df, target)

        bias_result = check_bias(df, preds, sensitive, target)

        result = bias_result

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)