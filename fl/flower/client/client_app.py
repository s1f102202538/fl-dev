from flwr.client import ClientApp

from fl.flower.client.apps.flower_demo_client import FlowerDemoClient

# Flower ClientApp
app = ClientApp(
  client_fn=FlowerDemoClient.client_fn,
)
