import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

@app.route("/", methods = ['POST','GET'])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        if username == "user1" and password == "user1":
            return  render_template("Home.html")
        else:
            return render_template("login_fail.html")
    return render_template("login.html")
@app.route('/home', methods=['GET', 'POST'])

def home():
    return render_template("Home.html")

@app.route('/main', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save("static/" + filename)
        file = open("static/" + filename,"r")
        model = torch.jit.load('Transformer_predict_Covid.pt')
        model.eval()
        img = Image.open("static/" + filename)
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.5), (0.5)), # imagenet means
            T.RandomErasing(p=0.2, value='random')
        ])
        img = img.convert("RGB")
        img = transform(img).unsqueeze(0)
        out = model(img)
        clsidx = torch.argmax(out)
        pred = clsidx.item()
        if pred == 0:
            s = "Covid"
        else:
            s = "Normal"
        return(render_template("main.html", result=str(s)))
    else:
        return(render_template("main.html", result="pending"))

@app.route('/CNN', methods=['GET', 'POST'])
def CNN():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save("static/" + filename)
        file = open("static/" + filename,"r")
        model = load_model("CNN_ResNet50")
        test_image = image.load_img("static/" + filename, target_size = (100, 100))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        pred = model.predict(test_image)
        if pred[0][0] == 1:
            s = "Covid"
        else:
            s = "Normal"
        return(render_template("CNN.html", result=str(s)))
    else:
        return(render_template("CNN.html", result="pending"))
if __name__ == "__main__":
    app.run()
