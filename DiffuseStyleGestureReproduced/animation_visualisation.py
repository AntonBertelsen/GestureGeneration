import os
import threading
import time
import socket
import asyncio
import json
import math
import random
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import websockets
from IPython.display import display, HTML
import nest_asyncio
nest_asyncio.apply()

# Get or create a persistent state object
try:
    # Check if we're in IPython/Jupyter
    ipython = get_ipython()
    
    # Try to get existing state from user namespace
    if 'bvh_viewer_state' not in ipython.user_ns:
        ipython.user_ns['bvh_viewer_state'] = {
            'http_server_thread': None,
            'http_server_port': 8000,
            'websocket_thread': None,
            'websocket_port': None,
            'active_websocket_connections': set(),
            'visualization_initialized': False
        }
    
    # Get reference to state
    state = ipython.user_ns['bvh_viewer_state']
except:
    # Fallback if not in IPython - this will be recreated on each run
    state = {
        'http_server_thread': None,
        'http_server_port': 8000,
        'websocket_thread': None,
        'websocket_port': None,
        'active_websocket_connections': set(),
        'visualization_initialized': False
    }

class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging

def find_free_port(start_port, end_port=None):
    """Find a free port in a given range"""
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
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_http_server(port=8000):
    """Starts a simple HTTP server on the given port."""
    state['http_server_port'] = port
    
    # Allow socket reuse
    TCPServer.allow_reuse_address = True
    
    # Start a new server instance
    handler = QuietHTTPRequestHandler
    try:
        httpd = TCPServer(("", port), handler)
        print(f"HTTP Server running on http://localhost:{port}")
        httpd.serve_forever()
    except OSError as e:
        print(f"HTTP Server error: {e}")

# WebSocket handling
# To this:
async def websocket_handler(websocket):
    """Handle incoming WebSocket connections"""
    state['active_websocket_connections'].add(websocket)
    print(f"WebSocket client connected. Total connections: {len(state['active_websocket_connections'])}")
    try:
        await websocket.wait_closed()
    finally:
        state['active_websocket_connections'].remove(websocket)
        print(f"WebSocket client disconnected. Remaining: {len(state['active_websocket_connections'])}")

async def start_websocket_server_async(host="localhost", port=None):
    """Start the WebSocket server asynchronously"""
    if port is None:
        port = find_free_port(8700, 8800)
        
    try:
        server = await websockets.serve(websocket_handler, host, port)
        print(f"WebSocket server running on ws://{host}:{port}")
        state['websocket_port'] = port
        return server, port
    except OSError as e:
        print(f"WebSocket Server error: {e}")
        return None, None

def start_websocket_server(host="localhost", port=None):
    """Start the WebSocket server in a background thread"""
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

# Function to send animation frames to connected clients
def send_frame(frame_data):
    """
    Send animation frame to all connected WebSocket clients
    
    Args:
        frame_data: Dictionary with animation data, should contain a "joints" list
        with position and rotation data for each joint
    """
    if not state['active_websocket_connections']:
        print("No WebSocket clients connected. Frame not sent.")
        return
    
    # Use asyncio to send the frame
    message = json.dumps(frame_data)
    
    async def send_to_all():
        await asyncio.gather(*[
            connection.send(message) 
            for connection in state['active_websocket_connections']
        ], return_exceptions=True)
    
    # Now we can safely use the event loop with nest_asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_to_all())

# Initialize servers and display visualization
def init_visualization():
    """Initialize the visualization if not already running"""
    # If already initialized, just return the current WebSocket port
    if state['visualization_initialized'] and state['websocket_port']:
        print(f"Visualization already running.")
        print(f"HTTP Server: http://localhost:{state['http_server_port']}")
        print(f"WebSocket Server: ws://localhost:{state['websocket_port']}")

        # Display the viewer
        html_file_path = f"http://localhost:{state['http_server_port']}/bvh_tests/bvh_visualisation/bvh_viewer_4.html?wsport={state['websocket_port']}"
        print(f"Viewer URL: {html_file_path}")
        iframe_html = f'<iframe src="{html_file_path}" width="512" height="512" frameborder="0"></iframe><br>'
        display(HTML(iframe_html))
        return state['websocket_port']
    
    # Check if HTTP port 8000 is in use but not by our server
    if check_port_in_use(8000) and (state['http_server_thread'] is None or not state['http_server_thread'].is_alive()):
        print("Port 8000 is in use by another application. Finding alternative port...")
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
    time.sleep(1)
    
    # Display the viewer
    html_file_path = f"http://localhost:{state['http_server_port']}/bvh_tests/bvh_visualisation/bvh_viewer_4.html?wsport={state['websocket_port']}"
    iframe_html = f'<iframe src="{html_file_path}" width="512" height="512" frameborder="0"></iframe><br>'
    display(HTML(iframe_html))
    
    print(f"Animation viewer ready!")
    print(f"HTTP Server: {html_file_path}")
    print(f"WebSocket Server: ws://localhost:{state['websocket_port']}")
    print("Remember to click 'Start Streaming' button in the viewer.")
    
    state['visualization_initialized'] = True
    return state['websocket_port']