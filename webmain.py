from flask import Flask, render_template, Response, url_for, send_from_directory
from net_config import ip_addr
import cv2
import os
import time
import json
import logging

app = Flask(__name__)
app.logger.setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
data = {}
proc_text = ""

@app.route('/Bisindo/user120/')
def index():
    return render_template('index.html')

@app.route('/Bisindo/user120/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory('static', path, cache_timeout=-1)

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
def gen():
    while True:
        try:
            # Refresh data
            if os.path.isfile('static/Bisindo/user120/web_data.json'):
                with open('static/Bisindo/user120/web_data.json') as f:
                    data = json.load(f)
                    proc_text = ""
            else:
                proc_text = "Processing"
                
            if len(os.listdir("RGB_images")) > 5:
                for root, dirs, files in os.walk("RGB_images", topdown=False):
                    for name in files:
                        fileName = os.path.join(root, name)
                        break
                    break
                img = cv2.imread(fileName)
                frame = cv2.imencode('.jpg', img)[1].tobytes()
                os.remove(fileName)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            pass

if __name__ == '__main__':
    with open('static/Bisindo/user120/web_data.json') as f:
        data = json.load(f)
    app.run(host=ip_addr, debug=False, port=5000)