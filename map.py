from flask import Flask, render_template, url_for, request

app = Flask(__name__)

 

@app.route('/')
@app.route('/home')
def home():
    return render_template("bivouac.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
    mouse_pos = request.form['mouse_info']
    zoom_level = request.form['zoom_level']
    print("Output :" + mouse_pos, flush=True)
    print("Zoom level :" + zoom_level, flush=True)

    return render_template('bivouac.html')
    


if __name__ == "__main__":
    app.run(debug=True)