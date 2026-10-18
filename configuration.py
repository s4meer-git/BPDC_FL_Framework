import torch


class configuration:


    # setup_remote.ipynb starts with these defaults and passes its chosen values to main.py.
    def __init__(self):


        # Choose the dataset and method here; main.py builds the matching model automatically.
        self.dataset = "tinyimagenet"
        self.method = "fedavg"
        self.model = "resnet18"


        # Common settings from the experiment setup; every method in a comparison uses the same values.
        self.num_of_clients = 30
        self.rounds = 40
        self.local_epochs = 5
        self.batch_size = 64
        self.lr = 0.01
        self.momentum = 0.9
        self.dirichlet_alpha = 0.5
        self.seed = 42


        # Client accuracy uses test data with the client's own training class mix, as "the test set associated with client i".
        self.test_partition = "matched"


        # Turn this on to simulate selfish clients; 0.05 gives one selfish client among 30, the paper's main scenario.
        self.selfishness = False
        self.selfish_fraction = 0.05
        self.selfish_strength = 0.7
        self.selfish_start_round = 3


        # FairRFL uses Dq-FFL weighting and the paper's tau; None estimates L from the shared local SGD steps.
        self.fairrfl_q = 1.0
        self.fairrfl_tau = 2.5
        self.fairrfl_dynamic_q = True
        self.fairrfl_lipschitz = None


        # The authors warm up for 50 of 200 rounds; 10 keeps that quarter of a 40-round run before history search.
        self.fedcda_memory_k = 3
        self.fedcda_warmup = 10
        self.fedcda_batch_client_num = 5
        self.fedcda_clients_per_round = None
        self.fedcda_smoothness = 1.0


        # FedMut uses full participation here; CIFAR-10 keeps the existing gradient clipping safeguard.
        self.fedmut_radius = 4.0
        self.fedmut_acc_rate = 0.3
        self.fedmut_bound = 50
        self.fedmut_fraction = 1.0
        self.fedmut_cifar_grad_norm = 10.0


        # FCFL defaults from the authors' options.py; fraction 1.0 keeps every client training each round as in the setup.
        self.fcfl_alpha = 0.5
        self.fcfl_fraction = 1.0


        # None uses up to four JPEG-loading processes for Tiny ImageNet on CUDA; MNIST and CIFAR-10 load faster without them.
        self.loader_workers = None


        # Use CUDA when available, otherwise MPS on the Mac or CPU on machines without either GPU backend.
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.reset_run_state()


    # main.py clears these caches before each run; they hold results, not experiment settings.
    def reset_run_state(self):
        self.current_round = 0
        self.fairrfl_client_losses = {}
        self.fairrfl_previous_losses = {}
        self.fairrfl_detected_clients = []
        self.fairrfl_local_steps = {}
        self.fedcda_history = {}
        self.fedcda_loss_history = {}
        self.fedcda_client_losses = {}
        self.fedcda_choices = {}
        self.fedcda_search_order = []
        self.fedmut_population = []
        self.fedmut_client_slots = {}
        self.fcfl_queue = {}
        self.fcfl_weights = {}
        self.fcfl_global_accuracy = None
        self.fcfl_train_accuracies = {}
        self.selfish_client_ids = []
        self.selfish_previous_global = None
        self.selfish_previous_updates = {}
        self.selfish_previous_shares = {}


    # Receives the global model and one client from server.py and sends them to the selected local training method.
    def train_client(self, global_model, client):
        method_name = self.method.lower()

        if method_name == "fairrfl":
            from methods.fairrfl import train_fairrfl_client

            return train_fairrfl_client(global_model, client, self)

        elif method_name == "fedmut":
            from methods.fedmut import train_fedmut_client

            return train_fedmut_client(global_model, client, self)

        elif method_name == "fedcda":
            from methods.fedcda import train_fedcda_client

            return train_fedcda_client(global_model, client, self)

        elif method_name == "fcfl":
            from methods.fcfl import train_fcfl_client

            return train_fcfl_client(global_model, client, self)

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

        elif method_name == "fedcda":
            from methods.fedcda import fedcda

            return fedcda(client_weights, client_sizes, global_model, clients, self)

        elif method_name == "fedmut":
            from methods.fedmut import fedmut

            return fedmut(client_weights, client_sizes, global_model, clients, self)

        elif method_name == "fcfl":
            from methods.fcfl import fcfl

            return fcfl(client_weights, client_sizes, global_model, clients, self)


        raise ValueError("Unknown method name: " + self.method)
