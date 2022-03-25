from flask import Flask, render_template, url_for, request

app = Flask(__name__)

 

@app.route('/')
@app.route('/home')
def home():
    return render_template("bivouac.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        mouse_pos = request.form['mouse_info']
        zoom_level = request.form['zoom_level']
        features = request.form['features']
        print("Output :" + mouse_pos, flush=True)
        print("Zoom level :" + zoom_level, flush=True)
        print("Features :" + features, flush=True)
        feature_extraction(features)
        return render_template('bivouac.html')
    else:
        return render_template('bivouac.html')
    
def feature_extraction(features):
    print(type(features))

if __name__ == "__main__":
    app.run(debug=True)