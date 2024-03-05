from intensitynet.inference import classify_intensity
from aodnet.inference import dehaze_aod
from dwgan.inference import dehaze_dwgan
from PIL import Image

def main(image):
    intensity = classify_intensity(image)
    print(intensity)
    if intensity=="heavy":
        result, image = dehaze_dwgan(image)
    else:
        result, image = dehaze_aod(image)

    return result, image

if __name__ == "__main__":
    data_hazy = Image.open("15_hazy.png")
    main(data_hazy)