import torch
import torchvision
import torch.optim
from aodnet.model import dehaze_net
import numpy as np
from PIL import Image
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dehaze_net = dehaze_net().cuda()
dehaze_net.load_state_dict(torch.load('weights/dehazer.pth',map_location=device))

stream = cv2.VideoCapture("forest_-_56326 (540p).mp4")

i=0
while stream.isOpened() is True:
    try:
        print(i)
        i+=1
        _, f = stream.read()

        f = cv2.resize(f,(320,240))
        data_hazy = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        data_hazy = (np.asarray(data_hazy)/255.0)
        data_hazy = torch.from_numpy(data_hazy).float()
        data_hazy = data_hazy.permute(2,0,1)
        data_hazy = data_hazy.cuda().unsqueeze(0)

        with torch.no_grad():
            clean_image = dehaze_net(data_hazy)
            clean_image = dehaze_net(clean_image)

        torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "output_img.png")
        im = cv2.imread("output_img.png")
        cv2.imshow("Result",im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print("Error")