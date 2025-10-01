from flwr.client import ClientApp

from flower.client.apps.fed_avg_client import FedAvgClient
from flower.client.apps.fed_kd_client import FedKdClient
from flower.client.apps.fed_moon_client import FedMoonClient

# Flower ClientApp
app = ClientApp(
  client_fn=FedMoonClient.client_fn,
)
