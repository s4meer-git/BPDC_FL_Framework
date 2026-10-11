import math
import copy
import torch
import printer
from support_modules.batchnorm import merge_batchnorm_buffers


# Receives minibatch loss, gradient counts and server step settings from FairMedian functions; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("methods.fairmedian", event, config=config, **data)


# Use the model and client supplied by config.train_client() to collect the gradient from one minibatch.
def collect_fairmedian_gradient(global_model, client, config):


    # This client ID and the number of training samples in its loader; info passing to printer.py.
    report_progress(
        config,
        "gradient_started",
        client_id=client.client_id,
        training_samples=len(client.train_loader.dataset),
    )

    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(config.device)
    local_model.train()
    local_model.zero_grad()
    loss_function = torch.nn.CrossEntropyLoss()
    gradients = {}

    for images, labels in client.train_loader:
        images = images.to(config.device)
        labels = labels.to(config.device)
        predictions = local_model(images)
        loss = loss_function(predictions, labels)
        loss.backward()

        for parameter_name, parameter in local_model.named_parameters():
            if parameter.grad is None:
                gradient = torch.zeros_like(parameter)
            else:
                gradient = parameter.grad
            gradients[parameter_name] = gradient.detach().cpu().clone()


        # Client ID, minibatch loss, sample count and number of gradient parameters; info passing to printer.py.
        report_progress(
            config,
            "gradient_collected",
            client_id=client.client_id,
            loss=loss.item(),
            batch_samples=labels.size(0),
            parameter_count=len(gradients),
        )

        break

    if len(gradients) == 0:
        raise ValueError("FairMedian requires a training batch and model parameters.")


    # Send the running statistics learned from this minibatch along with its parameter gradients.
    for name, buffer in local_model.named_buffers():
        gradients[name] = buffer.detach().cpu().clone()

    no_of_samples = len(client.train_loader.dataset)
    return gradients, no_of_samples


# config.aggregate() passes client gradients here to add noise, take the median and update the global model.
def fairmedian(client_weights, client_sizes, global_model, clients, config):
    no_of_clients = len(client_weights)

    if no_of_clients == 0:
        raise ValueError("FairMedian requires at least one client.")
    if no_of_clients % 2 == 0:
        raise ValueError("FairMedian requires an odd number of clients.")
    if len(client_sizes) != no_of_clients:
        raise ValueError("Each client must have a sample count.")
    if len(clients) != no_of_clients:
        raise ValueError("Each set of weights must have a matching client.")
    for size in client_sizes:
        if size <= 0:
            raise ValueError("FairMedian requires positive client sample counts.")
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("FairMedian requires a finite, positive learning rate.")
    if not math.isfinite(config.fairmedian_noise_scale) or config.fairmedian_noise_scale < 0:
        raise ValueError("fairmedian_noise_scale must be finite and at least zero.")

    if config.fairmedian_server_lr is None:
        server_lr = config.lr
    else:
        server_lr = config.fairmedian_server_lr

    if not math.isfinite(server_lr) or server_lr <= 0:
        raise ValueError("fairmedian_server_lr must be finite and positive.")


    if config.fairmedian_random_generator is None:
        config.fairmedian_random_generator = torch.Generator(device="cpu")
        config.fairmedian_random_generator.manual_seed(config.seed)

    global_weights = global_model.state_dict()
    new_weights = {}
    for name, value in global_weights.items():
        new_weights[name] = value.detach().cpu().clone()

    for parameter_name, parameter in global_model.named_parameters():
        global_parameter = global_weights[parameter_name].detach().cpu()
        noisy_gradients = []

        for weights in client_weights:
            gradient = weights[parameter_name].detach().cpu()
            if gradient.shape != global_parameter.shape:
                raise ValueError("Client gradient shape does not match: " + parameter_name)
            if not torch.isfinite(gradient).all().item():
                raise ValueError("FairMedian received a non-finite client gradient.")

            if config.fairmedian_noise_scale > 0:
                noise = torch.randn(
                    gradient.shape,
                    generator=config.fairmedian_random_generator,
                    dtype=gradient.dtype
                )
                gradient = gradient + config.fairmedian_noise_scale * noise

            noisy_gradients.append(gradient)

        gradient_matrix = torch.stack(noisy_gradients)
        sorted_gradients = torch.sort(gradient_matrix, dim=0).values
        middle_index = no_of_clients // 2

        median_gradient = sorted_gradients[middle_index]

        new_parameter = global_parameter - server_lr * median_gradient
        new_weights[parameter_name] = new_parameter.to(global_parameter.dtype)


    # Combine local BatchNorm statistics without applying gradient noise or the median step to these buffers.
    merge_batchnorm_buffers(new_weights, global_model, client_weights, client_sizes)


    # Client IDs, sample counts, Gaussian noise scale and server learning rate; info passing to printer.py.
    report_progress(
        config,
        "fairmedian_aggregated",
        client_ids=[client.client_id for client in clients],
        client_count=no_of_clients,
        client_sizes=client_sizes,
        noise_scale=config.fairmedian_noise_scale,
        server_lr=server_lr,
    )

    return new_weights
