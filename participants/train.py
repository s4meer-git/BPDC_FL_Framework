import copy
import math
import torch
import printer
from support_modules.precision import autocast, grad_scaler


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
    clip_norm = None
    scaler = grad_scaler(config)


    # Apply this safeguard only to FedMut on CIFAR-10; the shared SGD settings stay the same.
    if config.method.lower() == "fedmut" and config.dataset.lower() == "cifar10":
        clip_norm = config.fedmut_cifar_grad_norm
        if clip_norm is not None and (not math.isfinite(clip_norm) or clip_norm <= 0):
            raise ValueError("fedmut_cifar_grad_norm must be positive and finite, or None.")

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


            # With mixed precision the forward pass runs in fp16; cross-entropy itself is still computed in fp32.
            with autocast(config):
                predictions = local_model(images)
                loss = loss_function(predictions, labels)
            batch_loss = loss.item()


            # Stop at the first invalid loss so this client's weights are not updated with NaN or infinity.
            if not math.isfinite(batch_loss):
                raise ValueError(
                    f"Non-finite training loss ({batch_loss}) for {config.method}: "
                    f"round {getattr(config, 'current_round', 0)}, "
                    f"client {client_id}, epoch {epoch + 1}, batch {no_of_batches + 1}. "
                    "Check the training settings and the client's starting model."
                )

            scaler.scale(loss).backward()


            # Bound unusually large CIFAR-10 FedMut gradients before they enter SGD's momentum buffer.
            gradient_data = {}
            if clip_norm is not None:
                scaler.unscale_(optimizer)


                # A scaled fp16 step can overflow on purpose while the scaler calibrates; the scaler then skips that step.
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        local_model.parameters(), clip_norm, error_if_nonfinite=not scaler.is_enabled(),
                    ).item()
                except RuntimeError as error:
                    raise ValueError(
                        f"FedMut gradient check failed (non-finite gradient): round {config.current_round}, "
                        f"client {client_id}, epoch {epoch + 1}, batch {no_of_batches + 1}."
                    ) from error
                gradient_data = {
                    "gradient_norm_before_clip": grad_norm,
                    "gradient_norm_limit": clip_norm,
                    "gradient_clipped": grad_norm > clip_norm,
                }

            scaler.step(optimizer)
            scaler.update()

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
                **gradient_data,
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
