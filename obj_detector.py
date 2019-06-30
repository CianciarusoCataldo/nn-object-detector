#!/usr/bin/env python
"""
MIT License
Copyright (c) 2016 Paul Kramme
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author: Cianciaruso Cataldo
"""

import os
import sys

"""Root directory of the app"""
execution_path=os.getcwd()

from flask import Flask, request, redirect
from flask_compress import Compress
sys.path.append(os.path.join(execution_path,"detector"))
from detector.detector_server import Detector_Server
import gc

"""
Create a Flask app and then apply optimizations, like gzip compression and
caching policy
"""
app = Flask(__name__)
Compress(app)

import tensorflow as tf
graph=tf.get_default_graph()
detector=Detector_Server()

@app.route('/js/<string>')
def send_js(string):
    """
    Handle javascript module request from client.

    :param string: javascript module name
    :type string: str
    :returns: javascript file path
    :rtype: str
    """
    return send_from_directory(os.path.join(os.path.join(execution_path,'static'),'js'), string)

@app.route('/img/<string>')
def send_img(string):
    """
    Handle image request from client.

    :param string: image file name
    :type string: str
    :returns: image file path
    :rtype: str
    """
    return send_from_directory(os.path.join(os.path.join(execution_path,'static'),'img'), string)

@app.route('/css/<string>')
def send_css(string):
    """
    Handle css file request from client.

    :param string: css file name
    :type string: str
    :returns: css file path
    :rtype: str
    """
    return send_from_directory(os.path.join(os.path.join(execution_path,'static'),'css'), string)


@app.route('/fonts/<string>')
def send_fonts(string):
    """
    Handle font file request from client.

    :param string: font file name
    :type string: str
    :returns: font file path
    :rtype: str
    """
    return send_from_directory(os.path.join(os.path.join(execution_path,'static'),'fonts'), string)


@app.route('/favicon.ico')
def send_ico():
    """
    Handle icon file request from client.
    :returns: icon file path
    :rtype: str
    """
    return app.send_static_file('favicon.ico')


@app.route('/',methods = ['GET'])
def root():
    """
    Handle index file request from client.

    :returns: index file path
    :rtype: str
    """
    return app.send_static_file('index.html')



@app.route('/<path:path>',methods = ['GET'])
def error(path):
    """
    Handle eventual internal error. You can optionally use your custom error
    page, just add it to 'static' folder and return it with
    'app.send_static_file' method.

    :param path: file path
    """
    if not(os.path.isfile(path)):
        return "Application error : "+path+" not exists."



#Delete this rule if you want to handle other paths for POST request
#in your website
@app.route('/<path>',methods = ['POST'])
def invalid_request(path):
    return "Invalid request form : "+path

@app.route('/',methods = ['POST'])
def post_req():
    print("Receiving")
    image=request.files['image']
    content=image.read()
    message=""
    with graph.as_default():
        detector.detect(content)
    gc.collect()
    message=detector.get_result()
    return message
     
if __name__ == "__main__":
    pass
