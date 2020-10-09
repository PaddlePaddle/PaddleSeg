from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, File2Image, Resize, Transpose, BGR2RGB, Normalize, Div
import sys
import cv2
from postprocess import SegPostprocess

client = Client()
client.load_client_config("serving_client/serving_client_conf.prototxt")
client.connect(["127.0.0.1:9494"])

preprocess = Sequential([
    File2Image(),
    Resize((512, 512), interpolation=cv2.INTER_LINEAR),
    Div(255.0),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False),
    Transpose((2, 0, 1))
])

postprocess = SegPostprocess(2)

filename = sys.argv[1]
im = preprocess(filename)
fetch_map = client.predict(feed={"image": im}, fetch=["transpose_1.tmp_0"])
fetch_map["filename"] = filename
result_png = postprocess(fetch_map)
