import copy
import math
import torch
import printer
from support_modules.fedmut_diagnostics import check_tensors


# Receives client, epoch, batch and loss values from train_local() before forwarding them; info passing to printer.py.
def report_progress(config, event, **data):
    printer.report("participants.train", event, config=config, **data)


# Clients.train() supplies the model, loader and settings used to train this client's local copy.
def train_local(global_model, train_loader, config, client_id=None):


    # Train a separate copy for this client so its local updates do not change the shared global model.
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(config.device)
    local_model.train()

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=config.lr,
        momentum=config.momentum
    )

    loss_function = torch.nn.CrossEntropyLoss()
    diagnose = config.method.lower() == "fedmut"

    for epoch in range(config.local_epochs):
        total_loss = 0.0
        no_of_batches = 0


        # The client ID and local epoch that is about to start; info passing to printer.py.
        report_progress(
            config,
            "epoch_started",
            client_id=client_id,
            epoch=epoch + 1,
        )

        for images, labels in train_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()


            # Temporary FedMut checks record the input and starting weights for this exact training batch.
            if diagnose:
                location = dict(client_id=client_id, epoch=epoch + 1, batch=no_of_batches + 1)
                check_tensors(config, "before_forward", local_model.state_dict(), **location)
                check_tensors(config, "input_batch", {"images": images, "labels": labels}, **location)

            predictions = local_model(images)

            if diagnose:
                outputs = {"predictions": predictions, **dict(local_model.named_buffers())}
                check_tensors(config, "after_forward", outputs, **location)

            loss = loss_function(predictions, labels)
            batch_loss = loss.item()

            if diagnose:
                check_tensors(config, "loss", {"loss": loss}, **location)


            # Stop at the first invalid loss so this client's weights are not updated with NaN or infinity.
            if not math.isfinite(batch_loss):
                raise ValueError(
                    f"Non-finite training loss ({batch_loss}) for {config.method}: "
                    f"round {getattr(config, 'current_round', 0)}, "
                    f"client {client_id}, epoch {epoch + 1}, batch {no_of_batches + 1}. "
                    "Check the training settings and the client's starting model."
                )

            loss.backward()


            # Temporary FedMut checks record gradients before SGD and weights and momentum immediately after it.
            if diagnose:
                gradients = {name: parameter.grad for name, parameter in local_model.named_parameters()}
                check_tensors(config, "after_backward", gradients, **location)

            optimizer.step()

            if diagnose:
                updated = dict(local_model.state_dict())
                for name, parameter in local_model.named_parameters():
                    state = optimizer.state.get(parameter, {})
                    updated[f"momentum.{name}"] = state.get("momentum_buffer")
                check_tensors(config, "after_optimizer", updated, **location)

            total_loss += batch_loss
            no_of_batches += 1


            # Client ID, epoch, batch loss, batch sample count and accumulated epoch loss; info passing to printer.py.
            report_progress(
                config,
                "batch_completed",
                client_id=client_id,
                epoch=epoch + 1,
                batch=no_of_batches,
                loss=batch_loss,
                total_loss=total_loss,
                average_loss=total_loss / no_of_batches,
                batch_samples=labels.size(0),
            )

        if no_of_batches > 0:
            average_loss = total_loss / no_of_batches
        else:
            average_loss = 0.0


        # Client ID, completed epoch, batch count and the average training loss; info passing to printer.py.
        report_progress(
            config,
            "epoch_completed",
            client_id=client_id,
            epoch=epoch + 1,
            batches=no_of_batches,
            total_loss=total_loss,
            average_loss=average_loss,
        )

    final_weights = {}
    for name, value in local_model.state_dict().items():
        final_weights[name] = value.detach().cpu().clone()
    return final_weights
