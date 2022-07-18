import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel

import apex.amp as amp
from utils.config import cfg
from attacks import attack_pgd, attack_pgd_on_both, attack_pgd_on_conf

def train(model, device, train_loader, optimizer, epoch, tfboard_writer,):
    model.train()
    Acc = 0.
    total_loss = 0.
    loss_dict = dict()
    acc_dict = dict()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = F.cross_entropy(logits, target)
        acc = (logits.max(1)[1] == target).sum().item() / target.size(0)

        loss.backward()
        optimizer.step()

        Acc += acc
        total_loss += loss.item()

        loss_dict['loss'] = loss.item()
        tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.batch_size + batch_idx)

        acc_dict['accuracy'] = acc
        tfboard_writer.add_scalars(
            'training accuracy',
            acc_dict,
            epoch * cfg.batch_size + batch_idx
        )

    print('Train Epoch: {} \t CE: {:.4f} Acc: {:.2f}%'.format(epoch, total_loss / (batch_idx + 1), 100 * Acc / (batch_idx + 1)))
    loss_dict['loss'] = total_loss / (batch_idx + 1)
    acc_dict['accuracy'] = 100 * Acc / (batch_idx + 1)
    return loss_dict, acc_dict

def train_causal_poison(model, device, train_loader, causal_opt, conf_opt, epoch, tfboard_writer):
    model.train()
    Acc = 0.
    Acc_conf = 0.
    Acc_causal = 0.
    all_conf_loss = 0.
    all_causal_loss = 0.
    all_do_loss = 0.

    # user-defined class functions cannot be run on the scenario of distributed training;
    if isinstance(model, DataParallel):
        model = model.module

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        causal_opt.zero_grad()
        conf_opt.zero_grad()

        x_s, x_v, x_v_att, _ = model.split_x(data, target)
        x_do = x_s + x_v_att

        x_do = torch.clamp(x_do, 0., 1.)

        # confounding branch 
        conf_rep = model.get_conf_rep(x_v)
        conf_pred = model.get_conf_pred(conf_rep)
        conf_loss = F.cross_entropy(conf_pred, target)

        # causal branch 
        causal_rep = model.get_causal_rep(x_s)
        do_rep = model.get_causal_rep(x_do)

        causal_pred = model.get_causal_pred(causal_rep)
        do_pred = model.get_causal_pred(do_rep)

        causal_loss = F.cross_entropy(causal_pred, target)
        do_loss = F.cross_entropy(do_pred, target)        
        total_loss = cfg.TRAIN.CAUSAL_REG * do_loss + causal_loss

        acc_conf = (conf_pred.max(1)[1] == target).sum().item() / target.size(0)
        acc = (do_pred.max(1)[1] == target).sum().item() / target.size(0)
        acc_causal = (causal_pred.max(1)[1] == target).sum().item() / target.size(0)

        conf_loss.backward()
        conf_opt.step()

        total_loss.backward()
        causal_opt.step()

        Acc += acc
        Acc_conf += acc_conf
        Acc_causal += acc_causal
        all_conf_loss += conf_loss.item()
        all_causal_loss += causal_loss.item()
        all_do_loss += do_loss.item()

        loss_dict = dict()
        loss_dict['conf_loss'] = conf_loss.item()
        loss_dict['causal_loss'] = causal_loss.item()
        loss_dict['do_loss'] = do_loss.item()

        tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.batch_size + batch_idx)

        accdict = dict()
        accdict['do acc'] = acc
        accdict['confounding acc'] = acc_conf
        accdict['causal acc'] = acc_causal
        tfboard_writer.add_scalars(
            'training accuracy',
            accdict,
            epoch * cfg.batch_size + batch_idx
        )

    print('Train Epoch: {} \t Conf loss: {:.4f} Causal loss: {:.4f} Do loss: {:.4f} Conf Acc: {:.2f}% Causal Acc: {:.2f}%'.format(epoch, 
            all_conf_loss / (batch_idx + 1), 
            all_causal_loss / (batch_idx + 1), 
            all_do_loss / (batch_idx + 1), 
            100 * Acc_conf / (batch_idx + 1),
            100 * Acc / (batch_idx + 1)))
    loss_dict = dict()
    loss_dict['conf_loss'] = all_conf_loss / (batch_idx + 1)
    loss_dict['causal_loss'] = all_causal_loss / (batch_idx + 1)
    loss_dict['do_loss'] = all_do_loss / (batch_idx + 1)

    accdict = dict()
    accdict['do acc'] = 100 * Acc / (batch_idx + 1)
    accdict['causal acc'] = 100 * Acc_causal / (batch_idx + 1)
    accdict['confounding acc'] = 100 * Acc_conf / (batch_idx + 1) 
    return loss_dict, accdict

def train_causal_adv(model, device, train_loader, causal_opt, conf_opt, epoch, tfboard_writer):
    model.train()
    Acc = 0.
    Acc_conf = 0.
    Acc_causal = 0.
    all_conf_loss = 0.
    all_causal_loss = 0.
    all_do_loss = 0.
    # user-defined class functions cannot be run on the scenario of distributed training;
    if isinstance(model, DataParallel):
        model = model.module    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # causal_opt.zero_grad()
        # conf_opt.zero_grad()

        x_s, x_v, x_v_att, _ = model.split_x(data, target)
        # perform gradient-based attacks on the intervened data: x_do
        # causal do intervention 
        x_do = x_s + x_v_att
        x_do = torch.clamp(x_do, 0., 1.)
        x_do = x_do.detach()
        X = {'fg': x_s, 'bg': x_v}

        delta_X = attack_pgd_on_both(model, X, target, BNeval=cfg.TRAIN.BNeval, norm_input=cfg.TRAIN.NORM)
        # build final adversarial examples
        if cfg.TRAIN.AT_MODE == 'trades':
            x_v_natural = x_v.detach().clone()
        x_do_adv = x_s + x_v_att + delta_X['fg']
        x_v += delta_X['bg']

        # add data from adversarial domain into memory buffer
        if cfg.MODEL.CONF_ADV:
            model.erb.add(x_v.detach())

        # confounding branch 
        conf_rep = model.get_conf_rep(x_v)
        conf_pred = model.get_conf_pred(conf_rep)

        if cfg.TRAIN.AT_MODE == 'trades':
            criterion_kl = nn.KLDivLoss(size_average=False)
            conf_rep_natural = model.get_conf_rep(x_v_natural)
            conf_pred_natural = model.get_conf_pred(conf_rep_natural)
            conf_loss_robust = (1.0 / data.size(0)) * criterion_kl(F.log_softmax(conf_pred, dim=1),
                                              F.softmax(conf_pred_natural, dim=1))
            conf_loss = F.cross_entropy(conf_pred_natural, target) + cfg.TRAIN.TRADES_BETA * conf_loss_robust
        else:
            conf_loss = F.cross_entropy(conf_pred, target)

        # causal branch 
        # TODO: send original clean example into causal branch 
        if cfg.TRAIN.CAUSAL_SURRO == 'clean':
            causal_rep = model.get_causal_rep(x_s)
        elif cfg.TRAIN.CAUSAL_SURRO == 'adv':
            causal_rep = model.get_causal_rep(x_s+delta_X['fg'])
        else:
            raise ValueError
        causal_pred = model.get_causal_pred(causal_rep)

        # causal branch with do intervention
        do_rep = model.get_causal_rep(x_do_adv)
        do_pred = model.get_causal_pred(do_rep)

        if cfg.TRAIN.AT_MODE == 'trades' and cfg.TRAIN.CAUSAL_SURRO == 'adv':
            causal_rep_natural = model.get_causal_rep(x_s)
            causal_pred_natural = model.get_causal_pred(causal_rep_natural)
            causal_loss_robust = (1.0 / data.size(0)) * criterion_kl(F.log_softmax(causal_pred, dim=1),
                                                              F.softmax(causal_pred_natural, dim=1))
            causal_loss = F.cross_entropy(causal_pred_natural, target) + cfg.TRAIN.TRADES_BETA * causal_loss_robust
        else:
            causal_loss = F.cross_entropy(causal_pred, target)
        if cfg.TRAIN.CAUSAL_MODE == 'trades':
            do_rep_natural = model.get_causal_rep(x_do)
            do_pred_natural = model.get_causal_pred(do_rep_natural)
            do_loss_robust = (1.0 / data.size(0)) * criterion_kl(F.log_softmax(do_pred, dim=1),
                                                              F.softmax(do_pred_natural, dim=1))
            do_loss = F.cross_entropy(do_pred_natural, target) + cfg.TRAIN.TRADES_BETA * do_loss_robust
        else:
            do_loss = F.cross_entropy(do_pred, target)
        total_loss = cfg.TRAIN.CAUSAL_REG * do_loss + causal_loss

        acc_conf = (conf_pred.max(1)[1] == target).sum().item() / target.size(0)
        acc = (do_pred.max(1)[1] == target).sum().item() / target.size(0)
        acc_causal = (causal_pred.max(1)[1] == target).sum().item() / target.size(0)

        # minimize the confounding loss on the confounding branch
        conf_opt.zero_grad()
        conf_loss.backward()
        conf_opt.step()

        # minimize the toal loss on the backbone and the causal branch 
        causal_opt.zero_grad()
        total_loss.backward()
        causal_opt.step()

        Acc += acc
        Acc_conf += acc_conf
        Acc_causal += acc_causal
        all_conf_loss += conf_loss.item()
        all_causal_loss += causal_loss.item()
        all_do_loss += do_loss.item()

        loss_dict = dict()
        loss_dict['conf_loss'] = conf_loss.item()
        loss_dict['causal_loss'] = causal_loss.item()
        loss_dict['do_loss'] = do_loss.item()

        tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.batch_size + batch_idx)

        accdict = dict()
        accdict['do acc'] = acc
        accdict['confounding acc'] = acc_conf
        accdict['causal acc'] = acc_causal
        tfboard_writer.add_scalars(
            'training accuracy',
            accdict,
            epoch * cfg.batch_size + batch_idx
        )
        # end_time = time.time()
        # print("batch {}".format(end_time - start_time))
    print('Train Epoch: {} \t Conf loss: {:.4f} Causal loss: {:.4f} Do loss: {:.4f} Conf Acc: {:.2f}% Causal Acc: {:.2f}% Do Acc: {:.2f}%'.format(epoch, 
            all_conf_loss / (batch_idx + 1), 
            all_causal_loss / (batch_idx + 1), 
            all_do_loss / (batch_idx + 1), 
            100 * Acc_conf / (batch_idx + 1),
            100 * Acc_causal / (batch_idx + 1),
            100 * Acc / (batch_idx + 1)))

    loss_dict = dict()
    loss_dict['conf_loss'] = all_conf_loss / (batch_idx + 1)
    loss_dict['causal_loss'] = all_causal_loss / (batch_idx + 1)
    loss_dict['do_loss'] = all_do_loss / (batch_idx + 1)

    accdict = dict()
    accdict['do acc'] = 100 * Acc / (batch_idx + 1)
    accdict['causal acc'] = 100 * Acc_causal / (batch_idx + 1)
    accdict['confounding acc'] = 100 * Acc_conf / (batch_idx + 1) 
    return loss_dict, accdict

def train_causal_attack(model, device, train_loader, causal_opt, conf_opt, epoch, tfboard_writer,):
    model.train()
    Acc = 0.
    Acc_conf = 0.
    Acc_causal = 0.
    all_conf_loss = 0.
    all_causal_loss = 0.
    all_do_loss = 0.
    # user-defined class functions cannot be run on the scenario of distributed training;
    if isinstance(model, DataParallel):
        model = model.module
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        causal_opt.zero_grad()
        conf_opt.zero_grad()

        x_s, x_v, x_v_att, _ = model.split_x(data, target)
        # x_v_att = x_v_att.detach()

        # perform gradient-based attacks on the intervened data: x_do
        # causal do intervention 
        x_do = x_s + x_v_att.detach()
        x_do = torch.clamp(x_do, 0., 1.)
        X = {'bg': x_v}

        delta_X = attack_pgd_on_conf(model, X, target, BNeval=cfg.TRAIN.BNeval)

        # build final adversarial examples
        if cfg.TRAIN.AT_MODE == 'trades':
            x_v_natural = x_v
        x_v += delta_X['bg']
        # add data from adversarial domain into memory buffer
        model.erb.add(x_v.detach())

        # confounding branch 
        conf_rep = model.get_conf_rep(x_v)
        conf_pred = model.get_conf_pred(conf_rep)
        if cfg.TRAIN.AT_MODE == 'trades':
            criterion_kl = nn.KLDivLoss(size_average=False)
            conf_rep_natural = model.get_conf_rep(x_v_natural)
            conf_pred_natural = model.get_conf_pred(conf_rep_natural)
            conf_loss_robust = (1.0 / data.size(0)) * criterion_kl(F.log_softmax(conf_pred, dim=1),
                                                                   F.softmax(conf_pred_natural, dim=1))
            conf_loss = F.cross_entropy(conf_pred_natural, target) + cfg.TRAIN.TRADES_BETA * conf_loss_robust
        else:
            conf_loss = F.cross_entropy(conf_pred, target)

        # causal branch 
        if cfg.TRAIN.CAUSAL_SURRO == 'clean':
            causal_rep = model.get_causal_rep(x_s)
        else:
            raise ValueError

        causal_pred = model.get_causal_pred(causal_rep)
        # causal branch with do intervention
        do_rep = model.get_causal_rep(x_do)
        do_pred = model.get_causal_pred(do_rep)

        causal_loss = F.cross_entropy(causal_pred, target)
        do_loss = F.cross_entropy(do_pred, target)        
        total_loss = cfg.TRAIN.CAUSAL_REG * do_loss + causal_loss

        acc_conf = (conf_pred.max(1)[1] == target).sum().item() / target.size(0)
        acc = (do_pred.max(1)[1] == target).sum().item() / target.size(0)
        acc_causal = (causal_pred.max(1)[1] == target).sum().item() / target.size(0)

        conf_opt.zero_grad()

        # minimize the confounding loss on the confounding branch
        conf_loss.backward()
        conf_opt.step()

        causal_opt.zero_grad()
        # minimize the toal loss on the backbone and the causal branch 
        total_loss.backward()
        causal_opt.step()

        Acc += acc
        Acc_conf += acc_conf
        Acc_causal += acc_causal
        all_conf_loss += conf_loss.item()
        all_causal_loss += causal_loss.item()
        all_do_loss += do_loss.item()

        loss_dict = dict()
        loss_dict['conf_loss'] = conf_loss.item()
        loss_dict['causal_loss'] = causal_loss.item()
        loss_dict['do_loss'] = do_loss.item()

        tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.batch_size + batch_idx)

        accdict = dict()
        accdict['do acc'] = acc
        accdict['confounding acc'] = acc_conf
        accdict['causal acc'] = acc_causal
        tfboard_writer.add_scalars(
            'training accuracy',
            accdict,
            epoch * cfg.batch_size + batch_idx
        )

    print('Train Epoch: {} \t Conf loss: {:.4f} Causal loss: {:.4f} Do loss: {:.4f} Conf Acc: {:.2f}% Causal Acc: {:.2f}%'.format(epoch, 
            all_conf_loss / (batch_idx + 1), 
            all_causal_loss / (batch_idx + 1), 
            all_do_loss / (batch_idx + 1), 
            100 * Acc_conf / (batch_idx + 1),
            100 * Acc / (batch_idx + 1)))
    loss_dict = dict()
    loss_dict['conf_loss'] = all_conf_loss / (batch_idx + 1)
    loss_dict['causal_loss'] = all_causal_loss / (batch_idx + 1)
    loss_dict['do_loss'] = all_do_loss / (batch_idx + 1)

    accdict = dict()
    accdict['do acc'] = 100 * Acc / (batch_idx + 1)
    accdict['causal acc'] = 100 * Acc_causal / (batch_idx + 1)
    accdict['confounding acc'] = 100 * Acc_conf / (batch_idx + 1) 
    return loss_dict, accdict

def train_adv(model, device, train_loader, optimizer, epoch, tfboard_writer,):
    model.train()
    Acc = 0.
    total_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        delta_adv = attack_pgd(model, data, target, BNeval=cfg.TRAIN.BNeval)

        if cfg.TRAIN.AT_MODE == 'trades':
            logits = model(data)
            loss_natural = F.cross_entropy(logits, target)
            criterion_kl = nn.KLDivLoss(size_average=False)
            loss_robust = (1.0 / data.size(0)) * criterion_kl(F.log_softmax(model(data + delta_adv), dim=1),
                                              F.softmax(model(data), dim=1))
            loss = loss_natural + cfg.TRAIN.TRADES_BETA * loss_robust
        else:
            logits = model(data + delta_adv)
            loss = F.cross_entropy(logits, target)
        acc = (logits.max(1)[1] == target).sum().item() / target.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Acc += acc
        total_loss += loss.item()

        loss_dict = dict()
        loss_dict['loss'] = loss.item()
        tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.batch_size + batch_idx)

        accdict = dict()
        accdict['accuracy'] = acc
        tfboard_writer.add_scalars(
            'training accuracy',
            accdict,
            epoch * cfg.batch_size + batch_idx
        )

    print('Train Epoch: {} \t CE: {:.4f} Acc: {:.2f}%'.format(epoch, total_loss / (batch_idx + 1), 100 * Acc / (batch_idx + 1)))
    loss_dict['loss'] = total_loss / (batch_idx + 1)
    accdict['accuracy'] = 100 * Acc / (batch_idx + 1)
    return loss_dict, accdict

def eval_adv(model, device, test_loader, epoch, tfboard_writer, prefix='Test', att='untar'):
    if isinstance(model, DataParallel):
        model = model.module

    model.eval()
    test_ce = 0
    ce_adv = 0
    correct = 0
    correct_adv = 0

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        delta_adv = attack_pgd(model, data, target, loss_type=att, eval=True)
        
        logits = model(data)
        pred = logits.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if att=='adap_m':
            logits_adv = model.forward_m(data + delta_adv)
        else:
            logits_adv = model(data + delta_adv)

        pred_adv = logits_adv.max(1, keepdim=True)[1]
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()

        ce_loss = F.cross_entropy(logits, target)
        test_ce += ce_loss.item()
        ce_adv_loss = F.cross_entropy(logits_adv, target)
        ce_adv += ce_adv_loss.item()

    test_ce /= len(test_loader)
    ce_adv /= len(test_loader)
    test_accuracy = correct / len(test_loader.dataset)
    robust_acc = correct_adv / len(test_loader.dataset)

    print('{}: CE: {:.4f} Acc: {:.2f}% Robust CE: {:.4f} {} Robust Acc: {:.2f}%'.format(prefix, test_ce, 100.* test_accuracy, ce_adv, att, 100. * robust_acc))

    loss_dict = dict()
    loss_dict['CE loss'] = test_ce
    loss_dict['robust CE loss'] = ce_adv
    tfboard_writer.add_scalars(
        'eval loss',
        loss_dict,
        (epoch + 1) * cfg.batch_size
    )

    accdict = dict()
    accdict['clean acc'] = test_accuracy
    accdict['{} robust acc'.format(att)] = robust_acc

    tfboard_writer.add_scalars(
        'eval accuracy',
        accdict,
        (epoch + 1) * cfg.batch_size
    )

    return loss_dict, accdict