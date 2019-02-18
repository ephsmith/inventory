from flask import Flask
from flask import request
from flask import render_template
from flask import Markup
import plc

app = Flask(__name__)

w = plc.Controller()

@app.route('/', methods=['POST', 'GET'])
def inventory_app():
    last_op = 'place'

    if request.method == 'POST':
        if request.form['operation'] == 'pick':
            w.pick(request.form['character'].encode('ascii', 'ignore'))
        if request.form['operation'] == 'place':
            w.place(request.form['character'].encode('ascii', 'ignore'))
        last_op = request.form['operation']
    inv = w.inventory
    stock = [(k, inv[k]) for k in sorted(inv)]
    tables = w.print_inventory(tablefmt='html')
    z0_table = Markup(tables[0])
    z1_table = Markup(tables[1])
    return render_template('index.html',
                           stock=stock,
                           last_op=last_op,
                           z0_table=z0_table,
                           z1_table=z1_table)
