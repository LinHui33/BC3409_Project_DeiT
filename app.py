import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
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
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="2"))

if __name__ == "__main__":
    app.run()

