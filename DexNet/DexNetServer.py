import numpy as np
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
import base64

from DexNet.DexNetWrapper import DexNetWrapper
from constants import DexNet, Settings

import os

app = Flask(__name__)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '1024'

DexNetModel = DexNetWrapper()

@app.route("/process", methods=["POST"])
def process():

    payload = request.get_json()

    depth_im = np.frombuffer(
        base64.b64decode(payload["data"]),
        dtype=np.dtype(payload["dtype"])
    ).reshape(payload["shape"])
    print(depth_im.shape)

    grasps      = DexNetModel(depth_im, Settings.vis_DexNet)
    grasps_np   = DexNetModel.reformat_grasps(grasps)

    # print(grasps_np)1

    return jsonify(
        shape=list(grasps_np.shape),
        dtype=str(grasps_np.dtype),
        data=grasps_np.tobytes().hex()
    )

if __name__ == "__main__":
    DexNetModel.logger.info(f"⇢ Serving on {DexNet.url}")
    WSGIServer((DexNet.host, DexNet.port_num), app).serve_forever()