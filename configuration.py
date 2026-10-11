import torch


class configuration:


    # main.py uses these defaults when it creates the configuration for a new experiment.
    def __init__(self):


        # The earlier requested preset used 30 clients and 30 rounds; these are the settings used for the current runs.
        self.dataset = "tinyimagenet"
        self.model = "resnet18"


        # FairMedian requires an odd client count.
        self.num_of_clients = 3
        self.rounds = 3
        self.local_epochs = 3
        self.batch_size = 64
        self.lr = 0.01
        self.momentum = 0.9
        self.dirichlet_alpha = 0.5
        self.seed = 42


        # Choose "fedavg", "fairrfl", "fairmedian", "fedcda", or "fedmut".
        self.method = "fedavg"


        # FairRFL uses these fairness settings and keeps each client's losses between rounds.
        self.fairrfl_q = 1.0
        self.fairrfl_tau = 2.5
        self.fairrfl_client_losses = {}
        self.fairrfl_previous_losses = {}


        # FairMedian uses one minibatch per client per round, so local_epochs does not apply to this method.
        self.fairmedian_batch_size = 64


        # Add Gaussian noise to the gradients using this standard deviation in gradient units.
        self.fairmedian_noise_scale = 0.001


        # None uses self.lr; set a number to choose a separate server step.
        self.fairmedian_server_lr = None
        self.fairmedian_random_generator = None


        # FedCDA defaults from the authors' supplementary code.
        self.fedcda_memory_k = 3
        self.fedcda_warmup = 50
        self.fedcda_batch_client_num = 5


        # None uses all clients in this framework.
        self.fedcda_clients_per_round = None


        # Positive L in the paper's loss-plus-squared-divergence objective; this is a tunable coefficient.
        self.fedcda_smoothness = 1.0
        self.fedcda_history = {}
        self.fedcda_loss_history = {}
        self.fedcda_client_losses = {}
        self.fedcda_choices = {}
        self.fedcda_search_order = []


        # FedMut defaults from the authors' options.py.
        self.fedmut_radius = 4.0
        self.fedmut_acc_rate = 0.3
        self.fedmut_bound = 50


        # Use 0.1 for the authors' participation default.
        self.fedmut_fraction = 1.0
        self.fedmut_population = []
        self.fedmut_client_slots = {}
        self.current_round = 0


        # Use Colab's NVIDIA GPU when available, or the Apple GPU when running on this Mac.
        self.device = "cuda" if torch.cuda.is_available() else "mps"


        # if self.device == "mps" and not torch.backends.mps.is_available():
        #     raise RuntimeError("Apple GPU (MPS) is unavailable for this run.")


    # Uses the selected method in this config when main.py or printer.py asks for the training batch size.
    def get_training_batch_size(self):
        if self.method.lower() == "fairmedian":
            return self.fairmedian_batch_size

        return self.batch_size


    # Receives the global model and one client from server.py and sends them to the selected local training method.
    def train_client(self, global_model, client):
        method_name = self.method.lower()

        if method_name == "fairrfl":
            from methods.fairrfl import train_fairrfl_client

            return train_fairrfl_client(global_model, client, self)

        elif method_name == "fairmedian":
            from methods.fairmedian import collect_fairmedian_gradient

            return collect_fairmedian_gradient(global_model, client, self)

        elif method_name == "fedmut":
            from methods.fedmut import train_fedmut_client

            return train_fedmut_client(global_model, client, self)

        elif method_name == "fedcda":
            from methods.fedcda import train_fedcda_client

            return train_fedcda_client(global_model, client, self)

        return client.train(global_model, self)


    # Receives local weights, sample counts and clients from server.py and sends them to the selected aggregation method.
    def aggregate(self, client_weights, client_sizes, global_model, clients):
        method_name = self.method.lower()

        if method_name == "fedavg":
            from methods.fedavg import fedavg

            return fedavg(client_weights, client_sizes, config=self)

        elif method_name == "fairrfl":
            from methods.fairrfl import fairrfl

            return fairrfl(client_weights, client_sizes, global_model, clients, self)

        elif method_name == "fairmedian":
            from methods.fairmedian import fairmedian

            return fairmedian(client_weights, client_sizes, global_model, clients, self)

        elif method_name == "fedcda":
            from methods.fedcda import fedcda

            return fedcda(client_weights, client_sizes, global_model, clients, self)

        elif method_name == "fedmut":
            from methods.fedmut import fedmut

            return fedmut(client_weights, client_sizes, global_model, clients, self)


        raise ValueError("Unknown method name: " + self.method)
