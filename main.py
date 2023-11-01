from flask import *
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import scipy
import soundfile as sf
from pydub import AudioSegment
from collections import Counter

app = Flask(__name__)

def getPredictions(samples, sample_rate):
    peaks = scipy.signal.find_peaks(samples, height=0.1, distance = int(sample_rate/3))

    print ("No of peaks with scipy {}".format(len(peaks[0])))

    peaks_list = peaks[1]['peak_heights'].tolist()

    predictions = []

    for i, peak in enumerate(peaks_list) :
        sample_no_at_peak = peaks[0][i]

        start = int(sample_no_at_peak - (0.005 * sample_rate))
        end = int(sample_no_at_peak  + (0.035 * sample_rate))

        segment = samples [start : end]

        #sf.write('{}_{}.wav'.format(f_suf, i), segment, sample_rate, subtype='PCM_16')
        prediction = predict(segment, sample_rate)

        print("Prediction ", prediction)
        
        if prediction != 'Unsure' :
            predictions.append(prediction)

    return predictions


def predict(samples, sample_rate):
    #loaded_model = tf.keras.models.load_model("saved_models/brandisii_FCM.h5")
    loaded_model =  tf.keras.models.load_model("/home/foresthut/mysite/saved_models/brandisii_FCM.h5")

    print("In predict {}".format(len(samples)))
    mfccs_scaled_features = features_extractor(samples, sample_rate)


    print("In predict {}".format(mfccs_scaled_features.shape))
    mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)

    predicted_label = loaded_model.predict(mfccs_scaled_features)
    print("In predict {}".format(predicted_label))

    array = np.array(predicted_label) * 100

    classes_x = np.argmax(predicted_label,axis=1)

    if (np.max(array) < 90) :
        result = 'Unsure'
    else :
        result = classes_x[0]

    
    print("Class-wise weightage - " , array)
    print("prediction {}".format(result))

    return str(result)

def features_extractor(samples, sample_rate, n_mfccs=28):
    mfccs_features = librosa.feature.mfcc(y=samples, sr=sample_rate, hop_length= 256, win_length = 512, n_mfcc= n_mfccs, dtype=np.float32, n_fft=int(len(samples)/2))
    print(mfccs_features.shape)

    scaler = StandardScaler()
    mfccs_features_norm = scaler.fit(mfccs_features)
    mfccs_features_norm = scaler.transform(mfccs_features)

    scaled_features = np.mean(mfccs_features_norm, axis=1)
    return scaled_features


@app.route("/brandisii/v1.2", methods = ['POST'])
def brandisiiv1_2():
    if request.method != 'POST':
        return 'Invalid Request'

    fPCM = request.files['file']
    filename = fPCM.filename
    print("---------------------------------------------------------------------------------")

    samples, sample_rate = sf.read(fPCM, channels=1, samplerate=44100, format = 'RAW', subtype='PCM_16', endian = 'LITTLE')

    predictions = getPredictions(samples, sample_rate)

    finalResult = 404
    print("Predictions {}".format(predictions))
    if (len(predictions) > 1) :
        counter = Counter(predictions)
        finalResult = counter.most_common()[0][0]
    elif len(predictions) == 1 :
        finalResult = predictions[0]
    else :
        finalResult = 'Unsure'
    
    print("Final Result: {}".format(finalResult))
    return str(finalResult)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/madhvaii")
def madhvaii():
    return "silence is beautiful"

if __name__ == "__main__":
    app.run()
