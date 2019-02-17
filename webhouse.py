from flask import Flask
from flask import request
from flask import render_template
import plc

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def inventory_app():
    last_op = 'place'
    w = plc.Controller()
    if request.method == 'POST':
        if request.form['operation'] == 'pick':
            w.pick(request.form['character'])
        if request.form['operation'] == 'place':
            w.place(request.form['character'])
        last_op = request.form['operation']
    inv = w.inventory
    stock = [(k, inv[k]) for k in sorted(inv)]
    return render_template('index.html',
                           stock=stock,
                           last_op=last_op)
