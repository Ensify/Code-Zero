from intensitynet.inference import classify_intensity
from aodnet.inference import dehaze_aod
from dwgan.inference import dehaze_dwgan

def main(image):
    intensity = classify_intensity(image)
    if intensity=="heavy":
        result, image = dehaze_dwgan(image)
    else:
        result, image = dehaze_aod(image)

    print(f"Image saved in {result}")