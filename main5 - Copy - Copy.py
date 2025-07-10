import os
import uuid
import numpy as np  # Import numpy and alias it as np
import librosa
import noisereduce as nr
import soundfile as sf
from flask import Flask, render_template, request,jsonify,session,redirect,url_for
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import datetime
import pickle
from keras.models import load_model
from flask_mail import Mail, Message
import binascii

app = Flask(__name__)
app.secret_key = binascii.hexlify(os.urandom(24)).decode()


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'fatimaimran1382002@gmail.com'  # Use your actual Gmail address
app.config['MAIL_PASSWORD'] = 'qoos tzbu qhcf xirw'     # Use your generated App Password
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

y = ['crying', 'laugh', 'noise','silence']

model = load_model('CryingFinal.hdf5')
labelencoder = LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
moy = ['hungry', 'tired', 'discomfort','belly_pain','burping']


# Assuming 'y' contains your original string labels
le = LabelEncoder()
y_encoded = le.fit_transform(moy)  # Encode string labels to integers
n_mfcc = 40
n_fft = 1024  # setting the FFT size to 1024
hop_length = 10*16 # 25ms*16khz samples has been taken
win_length = 25*16 #25ms*16khz samples has been taken for window length
window = 'hann' #hann window used
n_chroma=12
n_mels=128
n_bands=7 #we are extracting the 7 features out of the spectral contrast
fmin=100
bins_per_ocatve=12


def extract_features(file_path):
    try:
        # Load audio file and extract features
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                            hop_length=hop_length, win_length=win_length).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                                     hop_length=hop_length, win_length=win_length,
                                                     n_mels=n_mels).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, n_fft=n_fft,
                                                             hop_length=hop_length, win_length=win_length,
                                                             n_bands=n_bands, fmin=fmin).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        return features
    except Exception as e:
        print(f"Error: Exception occurred in feature extraction - {str(e)}")
        return None

# Define the uploads folder
UPLOAD_FOLDER = 'records'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/page23', methods=['GET', 'POST'])
def page23():
    if 'email' not in session:  # Check if email is not in session
        if request.method == 'POST':
            email = request.form.get('email')
            name = request.form.get('name')  # Get the name as well
            if email and name:
                session['email'] = email  # Store the email in the session
                session['name'] = name    # Store the name in the session
                return redirect(url_for('home'))  # Redirect to home after setting the email
        return render_template('email1.html')  # Render the form to get email and name if it's not set
    return redirect(url_for('home'))  # Redirect to home if email is already in sessi
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("get_email.html")


      # Redirect to home if email is already in session

@app.route('/subscribe')
def subscribe():
    return render_template("email1.html")

@app.route('/home')
def home():
    email = session.get('email')  # Retrieve email from the session
    if email:
        send_alert(email)  # Call the function to send alert
        return render_template('app1.html')
    return redirect(url_for('subscribe'))  # Redirect to index if email is not found

def send_alert(email):
    # Logic to send an alert to the specified email
    print(f"Sending alert to {email}")
@app.route('/restart_session')
def restart_session():
    # Clear the session
    session.clear()
    # Redirect to the email input page
    return redirect(url_for('subscribe'))
@app.route('/page1')
def he():
    return render_template("loopaudio.html")

@app.route('/page2')
def real():
    return render_template("app1.html")



@app.route("/prediction", methods=["POST"])
def prediction():
    if 'img' not in request.files:
        return "No file part"
    img = request.files['img']
    if img.filename == '':
        return "No selected file"
    # Generate a unique filename
    filename = str(uuid.uuid4()) + ".wav"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(filepath)
    audio, sample_rate = librosa.load(filepath,res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    print(mfccs_scaled_features)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, mfccs_scaled_features.shape[0], 1)

    print(mfccs_scaled_features)
    print(mfccs_scaled_features.shape)
    predicted_label = model.predict(mfccs_scaled_features)
    print(predicted_label)
    predicted_class_index = np.argmax(predicted_label, axis=1)
    dataw = labelencoder.inverse_transform(predicted_class_index)
    data_list = dataw.tolist()
    predicted_class_string = ', '.join(data_list)
    print(predicted_class_string)
    predicted_confidences = predicted_label[0] * 100
    predicted_confidence_percentage = predicted_confidences[predicted_class_index[0]]

    # Print the prediction probabilities for all classes
    predictions = []
    for i, confidence in enumerate(predicted_confidences):
        class_name = labelencoder.inverse_transform([i])[0]
        confidence_percent = f"{confidence:.2f}%"
        predictions.append(f"{class_name}: {confidence_percent}")
    print(predictions)
    if predicted_class_string == "crying":
        with open('ReasonFinal.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        new_features = extract_features(filepath)

        if new_features is not None:
            # Reshape the feature array to match the model's expectations
            new_features = new_features.reshape(1, -1)  # Reshape to a 2D array with 1 row

            # Make predictions using the loaded model
            predicted_label = loaded_model.predict(new_features)


            probabilities = loaded_model.predict_proba(new_features)
            predicted_confidence_percentage = np.max(probabilities) * 100
            print(predicted_confidence_percentage)
            predictions = []
            for i, confidence in enumerate(probabilities[0]):
                class_name = le.inverse_transform([i])[0]
                confidence_percent = f"{round(confidence * 100)}%"  # Multiply by 100 and round to nearest whole number
                predictions.append(f"{class_name}: {confidence_percent}")
            predicted_label = le.inverse_transform(predicted_label)
            predicted_class_string=predicted_label[0]
            print(predicted_class_string)
            print(predictions)
            user_email = session.get('email')
            msg = Message(
                subject='Baby Cry Alert',
                sender='fatimaimran1382002@gmail.com',
                recipients=[user_email]
            )
            msg.body = f"Alert: The baby is crying due to {predicted_class_string}. Confidence: {predicted_confidence_percentage:.2f}% Other: {predictions}"
            mail.send(msg)

    return render_template("app2.html",data=predicted_class_string,confidence=predicted_confidence_percentage,predictions=predictions)

@app.route('/profile')
def profile():
    if 'name' in session and 'email' in session:
        name = session['name']
        email = session['email']
        return render_template('profile.html', name=name, email=email)
    else:
        print("Session variables missing!")  # Debugging line
        return redirect(url_for('index'))

    # Use labelencoder to transform the predicted class index back to the original class label
@app.route('/email', methods=["GET", "POST"])
def get_email():
    if request.method == "POST":
        email = request.form['email']
        session['user_email'] = email
        return redirect(url_for('index'))
    return render_template("email_form.html")

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

    y, sample_rate = librosa.load(filepath, sr=None)
    denoised_audio = nr.reduce_noise(y=y, sr=sample_rate)

    # === Step 3: Save the denoised audio ===
    denoised_filename = "denoised_" + filename
    denoised_filepath = os.path.join(app.config['UPLOAD_FOLDER'], denoised_filename)
    sf.write(denoised_filepath, denoised_audio, sample_rate)
    print(denoised_filename)

    audio, sample_rate = librosa.load(denoised_filepath, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    print(mfccs_scaled_features)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, mfccs_scaled_features.shape[0], 1)

    print(mfccs_scaled_features)
    print(mfccs_scaled_features.shape)
    predicted_label = model.predict(mfccs_scaled_features)
    print(predicted_label)
    predicted_class_index = np.argmax(predicted_label, axis=1)
    dataw = labelencoder.inverse_transform(predicted_class_index)
    data_list = dataw.tolist()
    predicted_class_string = ', '.join(data_list)
    print(predicted_class_string)
    predicted_confidences = predicted_label[0] * 100
    predicted_confidence_percentage = predicted_confidences[predicted_class_index[0]]

    # Print the prediction probabilities for all classes
    predictions = []
    for i, confidence in enumerate(predicted_confidences):
        class_name = labelencoder.inverse_transform([i])[0]
        confidence_percent = f"{confidence:.2f}%"
        predictions.append(f"{class_name}:   {confidence_percent}  ")
    print(predictions)
    if predicted_class_string == "crying":
        with open('ReasonFinal.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        new_features = extract_features(filepath)

        if new_features is not None:
            # Reshape the feature array to match the model's expectations
            new_features = new_features.reshape(1, -1)  # Reshape to a 2D array with 1 row

            # Make predictions using the loaded model
            predicted_label = loaded_model.predict(new_features)

            probabilities = loaded_model.predict_proba(new_features)
            predicted_confidence_percentage = np.max(probabilities) * 100


            print(predicted_confidence_percentage)
            predictions = []
            for i, confidence in enumerate(probabilities[0]):
                class_name = le.inverse_transform([i])[0]
                confidence_percent = f"{round(confidence * 100)}%"  # Multiply by 100 and round to nearest whole number
                predictions.append(f"{class_name}: {confidence_percent}")
            predicted_label = le.inverse_transform(predicted_label)
            predicted_class_string = predicted_label[0]
            print(predicted_class_string)
            print(predictions)
            user_email = session.get('email')
            msg = Message(
                subject='Baby Cry Alert',
                sender='fatimaimran1382002@gmail.com',
                recipients=[user_email]
            )
            msg.body = f"Alert: The baby is crying due to {predicted_class_string}. Confidence: {predicted_confidence_percentage:.2f}% Other: {predictions}"
            mail.send(msg)

    try:
        response_data = {
            "predicted_class": predicted_class_string,
            "confidence": float(predicted_confidence_percentage),  # Ensure it's a float
            "predictions": predictions  # List of strings should be serializable
        }
        print(response_data)  # Debugging print
        return jsonify(response_data)
    except Exception as e:
        print(f"Error occurred during JSON serialization: {str(e)}")
        return jsonify({"error": "An error occurred during JSON serialization"})
    # Use labelencoder to transform the predicted class index back to the original class label





if __name__ == "__main__":
    app.run(debug=True)
