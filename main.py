import os
from flask import Flask
from app import views

app = Flask(__name__)

app.add_url_rule('/', 'home', views.index)
app.add_url_rule('/app/', 'app', views.app)
app.add_url_rule('/app/gender/', 'gender',
                 views.genderApp, methods=['GET', 'POST'])

# DO NOT enable debug in production
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
