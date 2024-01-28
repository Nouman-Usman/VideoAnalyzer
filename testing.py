import google.generativeai as palm
import pyttsx3

from youtubeSearch import printResult

engine = pyttsx3.init()
API_KEY = 'AIzaSyDzj8yESjjCS6vWNIFAAnjaKVtjGTNsl8g'
palm.configure(api_key=API_KEY)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def recognizeSpeech():
    model_id = "models/text-bison-001"
    while True:
        try:
            text = input("Enter what you want:  ")
            prompt = text
            completion = palm.generate_text(
                model=model_id,
                prompt=prompt,
                temperature=0.99,
                max_output_tokens=800
            )
            text_to_speak = completion.result
            prompt2 = f'Kindly Extract title from the string that must be greater than three words: {text_to_speak}'
            completion2 = palm.generate_text(
                model=model_id,
                prompt=prompt2,
                temperature=0.99,
                max_output_tokens=800
            )
            printResult(completion2.result)
            # print(completion2.result)
            print(completion.result)
            # breakpoint()
            # engine.setProperty('rate', 160)
            # engine.say(text_to_speak)
            # engine.runAndWait()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    recognizeSpeech()
