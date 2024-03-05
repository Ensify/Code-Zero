import torch
import torch.nn as nn
from dwgan.model import fusion_net
from torchvision import transforms
from torchvision.utils import save_image as imwrite


def im_process(image):
    transform = transforms.Compose([transforms.ToTensor()])
    hazy = image.convert("RGB")
    print(hazy.size)
    if hazy.size!=(1600,1200):
        print("need to reshape ")
        hazy = hazy.resize((1600,1200))
    hazy = transform(hazy)
    hazy_up=hazy[:,0:1152,:]
    hazy_down=hazy[:,48:1200,:]
    return hazy_up,hazy_down

# --- Gpu device --- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = fusion_net()

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net)
net.load_state_dict(torch.load('weights/dehaze.pkl',torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
# --- Test --- #
print("-----DWGAN MODEL LOADED-----")

def dehaze_dwgan(image):
    hazy_up,hazy_down = im_process(image)
    with torch.no_grad():
        net.eval()
        hazy_up = hazy_up.to(device)
        hazy_down = hazy_down.to(device)
        frame_out_up = net(hazy_up.unsqueeze(0))
        frame_out_down = net(hazy_down.unsqueeze(0))
        frame_out = (torch.cat([frame_out_up[:, :, 0:600, :].permute(0, 2, 3, 1), frame_out_down[:, :, 552:, :].permute(0, 2, 3, 1)],1)).permute(0, 3, 1, 2)

    imwrite(frame_out, "result.png")
    return "result.png", frame_out