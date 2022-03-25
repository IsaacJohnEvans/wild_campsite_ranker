from flask import Flask, render_template, url_for, request

app = Flask(__name__)

 

@app.route('/')
@app.route('/home')
def home():
    return render_template("bivouac.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
    output = request.form['mouse_info']
    print("Output" + output, flush=True)

    return render_template('bivouac.html')
    


if __name__ == "__main__":
    app.run(debug=False)