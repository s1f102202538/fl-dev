from flwr.server import ServerApp

from flower.server.apps.fed_avg_server import FedAvgServer
from flower.server.apps.fed_kd_server import FedKDServer

# Create ServerApp
app = ServerApp(server_fn=FedKDServer.server_fn)
