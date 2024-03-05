from model import dehaze_net
import torch
import numpy as np 
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dehaze_net = dehaze_net().cuda()
dehaze_net.load_state_dict(torch.load('weights/dehazer.pth',map_location=device))
print("-----AOD NET MODEL LOADED-----")

def dehaze_aod(image):
    data_hazy = (np.asarray(data_hazy)/255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2,0,1)
    data_hazy = data_hazy.cuda().unsqueeze(0)    
    
    with torch.no_grad():
        clean_image = dehaze_net(data_hazy)

    torchvision.utils.save_image(clean_image, "live_out.png")

    return "live_out.png", clean_image