device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_resnet18(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_val_acc = -1.0

for epoch in range(num_epochs):
    # ---- train ----
    model.train()
    running_loss = 0.0
    train_correct, train_total = 0, 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("Loss/train", loss.item(), global_step)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100.0 * train_correct / train_total
    writer.add_scalar("Loss_epoch/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch) 

    # ---- val ----
    model.eval()
    val_loss_sum = 0.0
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss = val_loss_sum / len(val_loader.dataset)
    val_acc = 100.0 * val_correct / val_total
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    # consistent selection rule
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best.pt")

    print(f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} "
          f"| val_loss={val_loss:.4f} | val_acc={val_acc:.2f}")
