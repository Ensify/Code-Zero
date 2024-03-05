import websockets
from PIL import Image
import asyncio
import cv2
import base64

from main import main

port = 5000
print("Started server on port : ", port)

async def transmit(websocket, path):
    print("Client Connected !")
    try :
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            _, f = cap.read()
            data_hazy = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            image_path, image = main(data_hazy)

            im = cv2.imread(image_path)
            encoded = cv2.imencode('.jpg', im)[1]
            data = str(base64.b64encode(encoded))
            data = data[2:len(data)-1]
            await websocket.send(data)
        cap.release()

    except websockets.connection.ConnectionClosed as e:
        print("Client Disconnected !")
        cap.release()
    except:
        print("Someting went Wrong !")

start_server = websockets.serve(transmit,port=port,host="0.0.0.0")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()