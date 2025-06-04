import torch
import threading
import time
import socket
import asyncio
import json
import random
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import websockets
from IPython.display import display, HTML
import nest_asyncio
import base64
import io
import soundfile as sf
from utils.animation.skeleton import Skeleton

# Ensure that asyncio can run in Jupyter/IPython environments
nest_asyncio.apply()

default_state = {
    'http_server_thread': None,
    'http_server_port': 8000,
    'websocket_thread': None,
    'websocket_port': None,
    'active_websocket_connections': set(),
    'visualization_initialized': False
}

# Get or create a persistent state object
try:
    # Check if we're in IPython/Jupyter
    ipython = get_ipython()
    
    # Try to get existing state from user namespace
    if 'bvh_viewer_state' not in ipython.user_ns:
        ipython.user_ns['bvh_viewer_state'] = default_state.copy()
    
    # Get reference to state
    state = ipython.user_ns['bvh_viewer_state']
except:
    # Fallback if not in IPython - this will be recreated on each run
    state = default_state.copy()

class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging

def find_free_port(start_port, end_port=None):
    if end_port is None:
        end_port = start_port + 100
        
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    
    # If all ports in range are busy, try a random high port
    return random.randint(10000, 65000)

def check_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_http_server(port=8000):
    state['http_server_port'] = port
    
    # Allow socket reuse
    TCPServer.allow_reuse_address = True
    
    # Start a new server instance
    handler = QuietHTTPRequestHandler
    try:
        httpd = TCPServer(("", port), handler)
        httpd.serve_forever()
    except OSError as e:
        print(f"HTTP Server error: {e}")

async def websocket_handler(websocket):
    state['active_websocket_connections'].add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        state['active_websocket_connections'].remove(websocket)

async def start_websocket_server_async(host="localhost", port=None):
    if port is None:
        port = find_free_port(8700, 8800)
        
    try:
        server = await websockets.serve(websocket_handler, host, port)
        # print(f"WebSocket server running on ws://{host}:{port}")
        state['websocket_port'] = port
        return server, port
    except OSError as e:
        print(f"WebSocket Server error: {e}")
        return None, None

def start_websocket_server(host="localhost", port=None):
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start the WebSocket server
    server, port = loop.run_until_complete(start_websocket_server_async(host, port))
    state['websocket_port'] = port
    
    # Run the event loop
    try:
        loop.run_forever()
    except Exception as e:
        print(f"WebSocket loop error: {e}")
    finally:
        loop.close()

# Function to send data to connected clients
def send_message(message):
    if not state['active_websocket_connections']:
        # If no connections are active, we can't send the frame
        return
    
    async def send_to_all():
        await asyncio.gather(*[
            connection.send(json.dumps(message)) 
            for connection in state['active_websocket_connections']
        ], return_exceptions=True)
     
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_to_all())

# Initialize servers and display visualization
def init_visualization():
    # If already initialized, we simply display the viewer with the current state
    if state['visualization_initialized'] and state['websocket_port']:
        display_viewer()
    else:
        # We need to initialize the servers
        # Check if HTTP port 8000 is in use but not by our server
        if check_port_in_use(8000) and (state['http_server_thread'] is None or not state['http_server_thread'].is_alive()):
            # print("Port 8000 is in use by another application. Finding alternative port...")
            state['http_server_port'] = find_free_port(8001, 8100)
        else:
            state['http_server_port'] = 8000
        
        # Start HTTP server if not already running
        if state['http_server_thread'] is None or not state['http_server_thread'].is_alive():
            state['http_server_thread'] = threading.Thread(target=start_http_server, args=(state['http_server_port'],), daemon=True)
            state['http_server_thread'].start()
        
        # Start WebSocket server with a dynamically assigned port
        if state['websocket_thread'] is None or not state['websocket_thread'].is_alive():
            state['websocket_thread'] = threading.Thread(target=start_websocket_server, daemon=True)
            state['websocket_thread'].start()
    
        # Give servers time to start
        time.sleep(1.0)

        # Now we can display the viewer
        display_viewer()

        # Mark the visualization as initialized
        state['visualization_initialized'] = True

def display_viewer():
    html_file_path = f"http://localhost:{state['http_server_port']}/utils/animation/visualisation/new/animation_viewer.html?wsport={state['websocket_port']}"
    print(f"Viewer URL: {html_file_path}")
    
    iframe_html = f'<iframe src="{html_file_path}" width="512" height="512" frameborder="0"></iframe><br>'
    display(HTML(iframe_html))

def send_pose(pose: torch.Tensor, skeleton: Skeleton, pose_target_character: str = "default"):
    message = {}
    pose_data = {}
    pose_data["pose_target_character"] = pose_target_character
    pose_data["pose"] = skeleton.pose_to_websocket_format(pose)
    message["pose"] = pose_data
    send_message(message)

def send_debug_positions(debug_positions: torch.Tensor):
    
    # debug_positions is a list of positions that should be displayed as spheres
    # in the 3D viewer. It has the shape (N, 3) where N is the number of positions.
    debug_positions = debug_positions.cpu().numpy()
    
    message = {}

    # Initialize a list to hold the positions
    message["debug_positions"] = []

    # For each position in debug_positions, create a dictionary with the position data
    for pos_index in range(debug_positions.shape[0]):
        debug_position = debug_positions[pos_index]
        message["debug_positions"].append({
            "name": f"position_{pos_index}",
            "position": {
                "x": float(debug_position[0]),
                "y": float(debug_position[1]),
                "z": float(debug_position[2])
            }
        })
    send_message(message)

def send_debug_text(debug_text: str):
    # Add the debug text to the message
    message = {}
    message["debug_text"] = debug_text
    send_message(message)

def send_debug_tensor(debug_tensor: torch.Tensor, debug_tensor_name: str = ""):
    # Convert the tensor to a list and add it to the message    
    message = {}
    numpy_tensor = debug_tensor.cpu().numpy()
    message["debug_tensor"] = {
        "data": numpy_tensor.tolist(),
        "name": debug_tensor_name,
        "shape": list(numpy_tensor.shape),
        "min_value": float(numpy_tensor.min()),
        "max_value": float(numpy_tensor.max())
    }
    send_message(message)

def send_audio(audio, sample_rate=44100):
    try:
        message = {}
        audio_data = {}
        # Convert audio to proper WAV format with headers
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, sample_rate, format='WAV')
        audio_buffer.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        # Add audio chunk and format info to frame data
        audio_data["audio"] = audio_base64
        audio_data["audio_format"] = {
            "sampleRate": sample_rate,
            "format": "wav"
        }
        message["audio"] = audio_data
        send_message(message)
    except Exception as e:
        print(f"Error preparing audio data: {e}")

def send_character(name: str, position: tuple = (0, 0, 0), rotation: tuple = (0, 0, 0), color = 0xffffff):
    message = {}
    character_data = {
        "name": name,
        "position": {
            "x": position[0],
            "y": position[1],
            "z": position[2]
        },
        "rotation": rotation,
        "color": color
    }
    message["character"] = character_data
    send_message(message)