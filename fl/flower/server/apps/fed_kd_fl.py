class FedKDServer:
  @staticmethod
  def on_fit_config(server_round: int) -> object:
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
      lr /= 2
    return {"lr": lr}
