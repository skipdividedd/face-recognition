from dataset import data
from models import model
from facenet_pytorch import training
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from fastai.vision.all import *
from fastai.optimizer import OptimWrapper
import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook


device = torch.device('cpu')
print('Running on device: {}'.format(device))

def train(md_name, fast='False'):
    if md_name == 'resnet':
        train_loader, val_loader, test_loader = data('resnet')
        resnet = model('resnet')
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()),
                               lr=0.0005, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2, 4])
        loss_fn = nn.CrossEntropyLoss()
        epochs = 4
        metrics = {'fps': training.BatchTimer(),
                    'acc': training.accuracy}
        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        for epoch in range(epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            resnet.train()
            training.pass_epoch(resnet, loss_fn, train_loader, optimizer, scheduler,
                                batch_metrics=metrics, show_running=True, device=device,
                                writer=writer)

            resnet.eval()
            training.pass_epoch(resnet, loss_fn, val_loader,
                                batch_metrics=metrics, show_running=True, device=device,
                                writer=writer)

        writer.close()

        resnet.eval()
        training.pass_epoch(resnet, loss_fn, test_loader,
                            batch_metrics=metrics, show_running=True, device=device,
                            writer=writer)

        return resnet

    elif md_name == 'Triplet':
        train_loader, val_loader, test_loader = data('Triplet')
        model_triplet = model('Triplet')
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                               model_triplet.parameters()), lr=0.0001, weight_decay=1e-06)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2, 10, 15, 19])
        loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=1.1
        )

        class Logger(object):
            def __init__(self, mode, length, calculate_mean=False):
                self.mode = mode
                self.length = length
                self.calculate_mean = calculate_mean
                if self.calculate_mean:
                    self.fn = lambda x, i: x / (i + 1)
                else:
                    self.fn = lambda x, i: x

            def __call__(self, loss, i):
                track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
                loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
                print(track_str + loss_str + '   ', end='')
                if i + 1 == self.length:
                    print('')

        def pass_epoch_(model, loss_fn, loader, optimizer=None, scheduler=None,
                        show_running=True, device='cpu', writer=None, md=None):

            global i_batch
            mode = 'Train' if md == 'Train' else 'Valid'
            logger = Logger(mode, length=len(loader), calculate_mean=show_running)
            loss = 0
            for i_batch, (x, y, z, r) in enumerate(loader):
                anchor = model(x.cpu())
                positive = model(y.cpu())
                negative = model(z.cpu())
                loss_batch = loss_fn(anchor, positive,
                                     negative)
                if mode == 'Train':
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if writer is not None and mode == 'Train':
                    if writer.iteration % writer.interval == 0:
                        writer.add_scalars('loss',{mode: loss_batch.detach().cpu()},
                                           writer.iteration)
                    writer.iteration += 1

                loss_batch = loss_batch.cpu()
                loss += loss_batch
                if show_running:
                    logger(loss, i_batch)
                else:
                    logger(loss_batch, i_batch)

            if mode == 'Train' and scheduler is not None:
                scheduler.step()

            loss = loss / (i_batch + 1)

            if writer is not None and not mode == 'Train':
                writer.add_scalars('loss', {mode: loss}, writer.iteration)

            return loss

        epochs = 20
        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        for epoch in range(epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            model_triplet.train()
            pass_epoch_(model_triplet, loss_fn, train_loader, optimizer, scheduler,
                show_running=True, writer=writer,
                device=device, md='Train')

            model_triplet.eval()
            pass_epoch_(model_triplet, loss_fn, val_loader,
                        show_running=True, device=device,
                        writer=writer, md='Valid')
        writer.close()

        model_triplet.eval()
        pass_epoch_(model_triplet, loss_fn, test_loader,
                    show_running=True, device=device,
                    writer=writer, md='Valid')

        return model_triplet

    elif md_name == 'ArcFace':
        train_loader, val_loader, test_loader = data('ArcFace')
        resnet = model('ArcFace')
        if fast == 'True':
            dt = DataLoaders(train_loader, val_loader)
            loss = CrossEntropyLossFlat()
            opt_func = partial(OptimWrapper, opt=optim.Adam)
            learn = Learner(dt, resnet, metrics=accuracy, loss_func=loss, opt_func=opt_func)
            learn.fit_one_cycle(n_epoch=3, lr_max=1e-2, cbs=[ShowGraphCallback()])
        else:
            def fit_epoch(model, train_loader, criterion, optimizer):
                running_loss = 0.0
                running_corrects = 0
                processed_data = 0

                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    preds = torch.argmax(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    processed_data += inputs.size(0)

                train_loss = running_loss / processed_data
                train_acc = running_corrects.cpu().numpy() / processed_data
                return train_loss, train_acc

            def eval_epoch(model, val_loader, criterion):
                model.eval()
                running_loss = 0.0
                running_corrects = 0
                processed_size = 0

                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, 1)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    processed_size += inputs.size(0)
                val_loss = running_loss / processed_size
                val_acc = running_corrects.double() / processed_size
                return val_loss, val_acc

            def train(train_loader, val_loader, model, epochs):

                history = []
                best_acc = 0.0
                log_template = '\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
                val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}'

                with tqdm(desc="epoch", total=epochs) as pbar_outer:
                    opt = torch.optim.Adam(model.parameters())
                    criterion = nn.CrossEntropyLoss()
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

                    for epoch in range(epochs):
                        train_loss, train_acc = fit_epoch(model, train_loader, criterion, opt)
                        print("loss", train_loss)

                        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
                        history.append((train_loss, train_acc, val_loss, val_acc))
                        scheduler.step()  # val_loss
                        pbar_outer.update(1)
                        tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                                       v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
                        if val_acc > best_acc:
                            best_acc = val_acc
                            torch.save(model.state_dict(), 'Net.pth')
                            model.load_state_dict(torch.load('Net.pth'))
                return history

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()),
                                   lr=0.0005, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            history = train(train_loader, val_loader, model=resnet, epochs=10)

        return resnet


