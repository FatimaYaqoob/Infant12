
import os
import uuid
import numpy as np  # Import numpy and alias it as np
import librosa
from flask import Flask, render_template, request,jsonify
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import datetime
from keras.models import load_model

app = Flask(__name__)
y = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner', 'street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer', 'drilling']

model = load_model('audio_classification.hdf5')
labelencoder = LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

# Define the uploads folder
UPLOAD_FOLDER = 'records'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/save_audio", methods=["POST"])
def save_audio():
    audio_file = request.files['audio']
    print(audio_file)
    if audio_file.filename == '':
        return "No selected file"
    # Generate a unique filename
    filename = str(uuid.uuid4()) + ".wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)
    audio, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    print(mfccs_scaled_features)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, mfccs_scaled_features.shape[0], 1)

    print(mfccs_scaled_features)
    print(mfccs_scaled_features.shape)
    predicted_label = model.predict(mfccs_scaled_features)
    print(predicted_label)
    predicted_class_index = np.argmax(predicted_label, axis=1)
    predicted_class = labelencoder.inverse_transform(predicted_class_index)
    print(predicted_class)
    return render_template("app2.html",data=predicted_class)

    # Use labelencoder to transform the predicted class index back to the original class label





if __name__ == "__main__":
    app.run(debug=True)