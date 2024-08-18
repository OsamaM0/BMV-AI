import asyncio
import websockets
import numpy as np
from src.backbone import TFLiteModel, get_model
from src.landmarks_extraction import load_json_file 
from src.config import SEQ_LEN, THRESH_HOLD

s2p_map = {k.lower():v for k,v in load_json_file("src/sign_to_prediction_index_map.json").items()}
p2s_map = {v:k for k,v in load_json_file("src/sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

models_path = [
                './models/islr-fp16-192-8-seed42-fold0-best.h5',
]
models = [get_model() for _ in models_path]

# Load weights from the weights file.
for model, path in zip(models, models_path):
    model.load_weights(path)

tflite_keras_model = TFLiteModel(islr_models=models)
sequence_data = []

async def process_keypoints(websocket, path):
    res = []
    async for message in websocket:
        landmarks = np.frombuffer(message, dtype=np.float32).reshape(-1, 3)
        sequence_data.append(landmarks)
        
        sign = ""
        
        # Generate the prediction for the given sequence data.
        if len(sequence_data) % SEQ_LEN == 0:
            prediction = tflite_keras_model(np.array(sequence_data, dtype=np.float32))["outputs"]

            if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                sign = np.argmax(prediction.numpy(), axis=-1)
            
            sequence_data = sequence_data[10:]

        # Insert the sign in the result set if sign is not empty.
        if sign != "" and decoder(sign) not in res:
            res.insert(0, decoder(sign))
        
        await websocket.send(', '.join(str(x) for x in res))

start_server = websockets.serve(process_keypoints, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
