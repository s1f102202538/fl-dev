from flower.client.apps.fed_moon_client import FedMoonClient
from flower.server.apps.fed_kd_server import FedKDServer
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

client_app = ClientApp(client_fn=FedMoonClient.client_fn)

server_app = ServerApp(server_fn=FedKDServer.server_fn)

run_simulation(
  server_app=server_app,
  client_app=client_app,
  num_supernodes=5,
)
