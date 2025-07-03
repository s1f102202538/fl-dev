from flwr.server import ServerApp

from fl.flower.server.apps.flower_demo_server import FlowerDemoServer

# Create ServerApp
app = ServerApp(server_fn=FlowerDemoServer.server_fn)
