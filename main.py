from flask import *
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import sys
import wave
import scipy
import io
import soundfile as sf
from pydub import AudioSegment
from collections import Counter
from multiprocessing import Pool

app = Flask(__name__)

#loaded_model = tf.keras.models.load_model("/home/foresthut/mysite/saved_models/brandisii_FCM.h5")

#Used by v1.1
def createSegments(fName):
    samples, sample_rate = librosa.load(fName, sr = 44100)

    print ("The audio has a sample rate of {}Hz with {} samples.".format(sample_rate, samples.shape[0]))

    C= np.abs(librosa.cqt(y=samples, sr=sample_rate))
    o_env=librosa.onset.onset_strength(sr=sample_rate,S=librosa.amplitude_to_db(C,top_db=30))
    onset_env0 = librosa.onset.onset_detect(onset_envelope=o_env, sr=sample_rate,units='time', backtrack=True)
    print ("peaks (times) {} of size {}".format(onset_env0, len(onset_env0)))


    audio = AudioSegment.from_wav(fName)
    #print("Audio length (secs) {}".format(audio.duration_seconds))

    #j=0
    segments = []
    #Convert wav to audio_segment
    for i, peak in enumerate(onset_env0) :

        t1 = peak * 1000
        t2 = t1  + 100
        segment = audio[t1:t2]

        #print ("start time - {} : end time - {} for id {} with difference {}".format(t1 , t2 , i , t2-t1))

        sName = "/home/foresthut/mysite/tmp/segWav{}.wav".format(i)
        #print("Storing segment {}".format(sName))
        segment.export(sName, format="wav")
        segments.append(sName)
        #j=j+1

    print("Total {} segmenets created.".format(len(segments)))
    return segments

#Used by v1.1
def predict(fOut):
    loaded_model = tf.keras.models.load_model("/home/foresthut/mysite/saved_models/brandisii_FCM.h5")

    print("In predict {}".format(fOut))
    mfccs_scaled_features = features_extractor(fOut)


    print("In predict {}".format(mfccs_scaled_features.shape))
    mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)

    predicted_label = loaded_model.predict(mfccs_scaled_features)
    print("In predict {}".format(predicted_label))

    array = np.array(predicted_label) * 100

    classes_x = np.argmax(predicted_label,axis=1)
    print("Class-wise weightage - " , array)
    print("prediction {}".format(classes_x[0]))

    return str(classes_x[0])

#Used by v1.1
def features_extractor(f, n_mfccs=25):
    print("In features extractor {}".format(f))
    audio, sample_rate = librosa.load(f, sr=44100)
    print("In features extractor - segment loaded {}".format(audio.shape))

    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=1, n_mfcc= n_mfccs, dtype=np.float32)

    scaler = StandardScaler()
    mfccs_features_norm = scaler.fit(mfccs_features)
    mfccs_features_norm = scaler.transform(mfccs_features)

    scaled_features = np.mean(mfccs_features_norm, axis=1)
    return scaled_features


@app.route("/brandisii/v1.1", methods = ['POST'])
def brandisiiv1_1():
    if request.method != 'POST':
        return 'Invalid Request'

    fPCM = request.files['file']
    filename = fPCM.filename
    print("---------------------------------------------------------------------------------")

    fOutName = "/home/foresthut/mysite/tmp/{}.wav".format(filename.rsplit(".")[0])
    scipy.io.wavfile.write(fOutName, 44100, np.fromfile(fPCM, dtype=np.int16, count=-1))

    segments = []

    segments = createSegments(fOutName)

    print("Segments {}".format(segments))

    predictions = []

    #for segment in range(noOfSegments):
    #    segments.append("/home/foresthut/mysite/tmp/segWav{}.wav".format(segment))

    #with Pool(5) as p:
    #    predictions = p.map(predict, segments)

    for segment in range(len(segments)):
       predictions.append(predict("/home/foresthut/mysite/tmp/segWav{}.wav".format(segment)))

    finalResult = 404

    print("Predictions {}".format(predictions))

    if (len(predictions) > 1) :
        counter = Counter(predictions)
        finalResult = counter.most_common()[0][0]
    else:
        finalResult = predictions[0]

    print("Final Result: {}".format(finalResult))

    return "Predictions {} and Final Result {}".format(predictions, str(finalResult))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/madhvaii")
def madhvaii():
    return "silence is beautiful"


if __name__ == "__main__":
    app.run()