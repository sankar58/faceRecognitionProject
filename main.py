from flask import Flask
from app import views

app = Flask(__name__)

app.add_url_rule("/", "home", views.index)
app.add_url_rule("/app/", "app", views.app)
app.add_url_rule(
    "/app/gender/",
    "gender",
    views.genderApp,
    methods=["GET", "POST"]
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
