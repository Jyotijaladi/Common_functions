from gtts import gTTS
import playsound

def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    playsound.playsound(filename)
