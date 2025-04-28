import base64

audio_bytes = chunk.tobytes()
audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

message = {
    "frame_data": frame_data,  # existing animation info
    "audio_chunk": audio_b64,
}
websocket.send(json.dumps(message))
