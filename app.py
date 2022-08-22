from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="Image Path")
    args = vars(ap.parse_args())
    img_path = args['image']
    
    max_length = 32

    img = request.files['file1']

    img.save('static/file.jpg')
    img_path = 'static/file.jpg'
    print("="*50)
    print("IMAGE SAVED")

    tokenizer = load(open('tokenizer.p',"rb"))
    model = load_model('models\model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    def extract_features(filename, model):
            try:
                image = Image.open(filename)
                
            except:
                print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
            image = image.resize((299,299))
            image = np.array(image)
            # for images that has 4 channels, we convert them into 3 channels
            if image.shape[2] == 4: 
                image = image[..., :3]
            image = np.expand_dims(image, axis=0)
            image = image/127.5
            image = image - 1.0
            feature = model.predict(image)
            return feature

    def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None


    def generate_desc(model, tokenizer, photo, max_length):
        in_text = 'start'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = model.predict([photo,sequence], verbose=0)
            pred = np.argmax(pred)
            word = word_for_id(pred, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text


    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    # print("\n\n")
    # print(description)
    # plt.imshow(img)


    #Remove start and end
    query = description
    stopwords = ['start','end']
    querywords = query.split()

    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    print(result)
    return render_template('after.html', data=result)

if __name__ == "__main__":
    app.run(debug=True)


