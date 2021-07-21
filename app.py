import threading
from functools import partial
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import requests
import mtcnn
from torch import nn
from resnet50_ft_dag import resnet50_ft_dag

class MainScreen(Screen):
    pass

class Manager(ScreenManager):
    pass

Builder.load_string('''
<MainScreen>:
    FloatLayout:
        Label:
            text: "RECONHECEDOR DE FACES"
            pos_hint: {"x":0.0, "y":0.85}
            size_hint: 1.0, 0.2
        Image:
            id: vid
            size_hint: 1, 0.8
            allow_stretch: True  # allow the video image to be scaled
            keep_ratio: True  # keep the aspect ratio so people don't look squashed
            pos_hint: {'center_x':0.5, 'top':0.9}
        Button:
            id:btnExit
            size_hint: .2, .05
            pos_hint: {'center_x': .5, 'center_y': .05}
            text:"Fechar"
            on_press: root.fechar()
''')

class Main(App):

    def build(self):
        self.title = 'Interface com Kivy'
        threading.Thread(target=self.facedetect, daemon=True).start()
        sm = ScreenManager()
        self.main_screen = MainScreen()
        sm.add_widget(self.main_screen)
        return sm

    def facedetect(self):
        metric = nn.L1Loss()
        users_path = os.path.join(os.getcwd(), 'pictures') 
        self.do_vid = True
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cam = cv2.VideoCapture(0)
        while (self.do_vid):
            ret, frame = cam.read()
            pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(
                pixels,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            result = self.analisa_nova_imagem(pixels, users_path, metric)
            for (x, y, w, h) in faces:                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            Clock.schedule_once(partial(self.display_frame, frame))

    def display_frame(self, frame, dt):        
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.main_screen.ids.vid.texture = texture

    def fechar(self):
        self.parent.remove_widget(self)

        # Baixe o modelo treinado para reconhecimento de faces
    def download_model(self):    
        if not os.path.isfile('resnet50_ft_dag.pth'):
            weights_path = 'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_ft_dag.pth'
            r = requests.get(weights_path, allow_redirects=True)
            open('resnet50_ft_dag.pth', 'wb').write(r.content)
        model = resnet50_ft_dag('resnet50_ft_dag.pth')
        return (model)

    # Função para extração de características a partir de uma imagem
    def extract_features(self, pixels):
        detector = mtcnn.MTCNN()
        model = self.download_model()
        #print(model)
        pixels = np.array(pixels, dtype=np.uint8)    
        # Recortando a face
        faces = detector.detect_faces(pixels)
        #print(faces)
        x, y, width, height = faces[0]['box']
        face = pixels[y:y+height, x:x+width]    
        # Plot da imagem e da face recortada
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(pixels)
        axs[1].imshow(face)    
        # extrai caracteristicas de alto nível
        face = Image.fromarray(face, 'RGB')
        face = transforms.Resize( (224, 224) )(face)
        face = transforms.ToTensor()(face).unsqueeze(0)    
        class_, feature = model(face)     
        return feature.detach().cpu().data.squeeze()

    # Função para guardar as features de referência de um dado usuário
    def registra_usuario(self, usuario, metric, users_path):        
        file = os.path.isfile(users_path+'/'+'frank'+'/'+'referencia.npz')
        if not file:
            user_path = os.path.join(users_path, usuario)
            all_features = []        
            for img in os.listdir(user_path):
                if img[-3:] != 'jpg': continue            
                pixels = Image.open(os.path.join(user_path, img))
                feature = self.extract_features(pixels)        
                all_features.append(feature)            
            all_losses = []
            for k in range(len(all_features)):
                for j, feat in enumerate(all_features):
                    if k == j: continue
                    all_losses.append(metric(all_features[k], feat) )        
            all_losses = np.asarray(all_losses)
            print(np.mean(all_losses), np.std(all_losses))        
            all_features = np.asarray([feat.numpy() for feat in all_features])
            np.savez_compressed(os.path.join(user_path, 'referencia'), all_feats=all_features, 
                                                            mean=np.mean(all_losses),
                                                            std=np.std(all_losses))

    #Analisa imagens historicas
    def analisa_banco_imagens(self, users_path, metric):
        for user in os.listdir(users_path):
            #print(user.capitalize())
            self.registra_usuario(user, metric, users_path)

    #Analisa nova imagem
    def analisa_nova_imagem(self, pixels, users_path, metric):        
        feature = self.extract_features( pixels )
        # Compara feature da nova imagem com as referências previamente armazenadas
        reconhecido = False
        result = ""
        for user in os.listdir(users_path):    
            referencia = np.load(os.path.join(users_path, user, 'referencia.npz'))
            all_features = referencia['all_feats']
            mean = referencia['mean']
            std  = referencia['std']    
            all_dist = []
            for feat in all_features:
                all_dist.append(metric(feature, torch.from_numpy(feat) ))    
            if abs( np.mean(all_dist) - mean ) < std:
                result = user.capitalize()
                reconhecido = True        
        if not reconhecido:
            result = ('Desconhecido')
        return result

if __name__ == '__main__':
    Main().run()