from flower.server.apps.fed_kd_server import FedKDServer
from flwr.server import ServerApp

# Create ServerApp
app = ServerApp(server_fn=FedKDServer.server_fn)
