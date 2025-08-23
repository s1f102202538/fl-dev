from flower.client.apps.fed_kd_client import FedKDClient
from flwr.client import ClientApp

# Flower ClientApp
app = ClientApp(
  client_fn=FedKDClient.client_fn,
)
