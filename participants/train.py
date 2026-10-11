import copy
import torch
import printer


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

            predictions = local_model(images)
            loss = loss_function(predictions, labels)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
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
