from participants.train import train_local
import printer


# Receives client IDs and sample counts from Clients.train() before forwarding them; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("participants.client", event, config=config, **data)


class Clients:


    # Receives the client ID and train/test loaders built by create_clients() in main.py.
    def __init__(self, client_id, train_loader, test_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader


    # Receives a starting model and config from the training method, then returns local weights and the sample count.
    def train(self, global_model, config):
        training_samples = len(self.train_loader.dataset)


        # This client ID and the number of training samples in its loader; info passing to printer.py.
        report_progress(
            config,
            "client_started",
            client_id=self.client_id,
            training_samples=training_samples,
        )

        updated_weights = train_local(
            global_model,
            self.train_loader,
            config,
            client_id=self.client_id
        )


        # The trained client ID and number of samples used for its local training; info passing to printer.py.
        report_progress(
            config,
            "client_completed",
            client_id=self.client_id,
            training_samples=training_samples,
        )

        return updated_weights, training_samples
