from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from tensorflow import keras
import cv2
import numpy as np
from keras.preprocessing import image

classes = ['apple', 'banana', 'beetroot', 'bell paper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli paper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']


class MenuWindow(Screen):
    pass

class CameraWindow(Screen):
    def capture(self):
        camera = self.ids['camera']
        camera.export_to_png("photo.png")

class ValidationWindow(Screen):
    clas_name = ''
    def recognize(self):
        model = keras.models.load_model('model.h5')
        img = keras.utils.load_img('photo.png', target_size=(160, 160))
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        print(x[0])
        pred = model.predict(x)
        ans = np.argmax(pred, axis=1)
        content = Button(text='Тут будет описание распознанных продуктов ')
        popup = Popup(title='На фотографии изображен продукт ' + classes[ans[0]],content = content,
                      auto_dismiss=False)

        content.bind(on_press=popup.dismiss)
        popup.open()


class InstructionWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass


kv = Builder.load_file('style.kv')

class NeyronApp(App):
    def build(self):
        return kv

if __name__ == '__main__':
	NeyronApp().run()