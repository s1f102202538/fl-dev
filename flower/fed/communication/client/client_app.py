from flwr.client import ClientApp

from .apps.fed_avg_client import FedAvgClient

# Flower ClientApp
app = ClientApp(
  client_fn=FedAvgClient.client_fn,
)
