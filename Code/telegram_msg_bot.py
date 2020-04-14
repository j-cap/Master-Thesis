

import urllib

def send_TG_msg(text):
    if text.find(" ") > 0:
        text = text.replace(" ", "+")
    token = "1200420075:AAEqUONzhZEfBE6yPCGjWuftiHQANW0d_QI"
    chatID = "-458410684"
    url = "https://api.telegram.org/bot"+token+"/sendMessage?chat_id="+chatID+"&text="+text
    # print(url)
    urllib.request.urlopen(url)
