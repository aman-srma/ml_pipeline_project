from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])

def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    
    else:
        data = CustomClass(
            age = int(request.form.get("age")),
            workclass = int(request.form.get("workclass")),
            education_num = int(request.form.get("education-num")),
            marital_status = int(request.form.get("marital-status")),
            occupation = int(request.form.get("occupation")),
            relationship = int(request.form.get("relationship")),
            race = int(request.form.get("race")),
            gender = int(request.form.get("gender")),
            capital_gain = int(request.form.get("capital-gain")),
            capital_loss = int(request.form.get("capital-loss")),
            hours_per_week = int(request.form.get("hours-per-week")),
            native_country = int(request.form.get("native-country")),
        )

    final_data = data.get_data_DataFrame()
    pipeline_prediction = PredictionPipeline()
    pred = pipeline_prediction.predict(final_data)

    result = pred

    if result == 0:
        return render_template("results.html", final_result = "Your Yearly Income Is Less Than Equal To $50k: {}".format(result))
    elif result == 1:
        return render_template("results.html", final_result = "Your Yearly Income Is More Than $50k: {}".format(result))
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
