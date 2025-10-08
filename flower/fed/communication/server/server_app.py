from flwr.server import ServerApp

from .apps.fed_avg_server import FedAvgServer

# Create ServerApp
app = ServerApp(server_fn=FedAvgServer.server_fn)
