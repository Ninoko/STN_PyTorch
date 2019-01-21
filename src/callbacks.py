def after_batch(id, img, loss, cumulative_loss, img_num, mode='Training'):
    img_num += len(img)
    cumulative_loss += loss.item()
    print('{} - Loss'.format(mode), id, loss.item())
    id += 1

    return id, cumulative_loss, img_num


def after_epoch(epoch_id, cumulative_loss, img_num, mode='Training'):
    print('{} - Loss per epoch'.format(mode), epoch_id, cumulative_loss / img_num)