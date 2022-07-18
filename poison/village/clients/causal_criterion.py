import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class causal_perturb:
    def __init__(self, args):
        self.beta = args.causal_beta
        self.loss_type = args.causal_loss_type
        self.class_mu = torch.zeros(1)

    def update_mu(self, class_mu):
        self.class_mu = class_mu

    def run(self, causal_model, inputs, labels, new_label=None):
        if self.loss_type == 's2zero':
            s_rep = causal_model.get_causal_rep(inputs)
            loss = torch.norm(s_rep,p=2)
        elif self.loss_type == 's2random':
            s_rep = causal_model.get_causal_rep(inputs)
            s_rep = F.sigmoid(s_rep)
            rand_noise = torch.rand_like(s_rep)
            loss = torch.norm(s_rep - rand_noise, p=2)
        elif self.loss_type == 'perturb_s':
            s_rep = causal_model.get_causal_rep(inputs)
            if new_label is not None:
                mu = self.class_mu[new_label]
                loss = torch.norm(s_rep - mu, p=2)
            else:
                mu = self.class_mu[labels]
                loss = (-1) * torch.norm(s_rep - mu, p=2)
        elif self.loss_type == 'perturb_s_output':
            if new_label is not None:
                output = causal_model(inputs)
                loss = F.cross_entropy(output, new_label)
            else:
                output = causal_model(inputs)
                loss = (-1) * F.cross_entropy(output, labels)

        elif self.loss_type == 'perturb_v':
            x_s, x_v, x_v_att = causal_model.split_x(inputs, labels, eval=True)
            v_rep = causal_model.get_conf_rep(x_v)
            if new_label is not None:
                mu = self.class_mu[new_label]
                loss = torch.norm(v_rep - mu, p=2)
            else:
                mu = self.class_mu[labels]
                loss = (-1) * torch.norm(v_rep - mu, p=2)

        elif self.loss_type == 'perturb_v_output':
            x_s, x_v, x_v_att = causal_model.split_x(inputs, labels, eval=True)
            v_rep = causal_model.get_conf_rep(x_v)
            v_pred = causal_model.get_conf_pred(v_rep)
            if new_label is not None:
                loss = F.cross_entropy(v_pred, new_label)
            else:
                loss = (-1) * F.cross_entropy(v_pred, labels)

        elif self.loss_type == 'perturb_v_output_min':
            x_s, x_v, x_v_att = causal_model.split_x(inputs, labels, eval=True)
            v_rep = causal_model.get_conf_rep(x_v)
            v_pred = causal_model.get_conf_pred(v_rep)
            loss = F.cross_entropy(v_pred, labels)

        elif self.loss_type == 'perturb_s_output_min':
            output = causal_model(inputs)
            loss = F.cross_entropy(output, labels)

        elif self.loss_type == 'perturb_v_and_s2random':
            x_s, x_v, x_v_att = causal_model.split_x(inputs, labels, eval=True)
            v_rep = causal_model.get_conf_rep(x_v)
            s_rep = causal_model.get_causal_rep(x_s)
            s_rep = F.sigmoid(s_rep)

            rand_noise = torch.rand_like(s_rep)
            if new_label is not None:
                mu = self.class_mu[new_label]

            else:
                raise ValueError("new_label can not be None")
            loss = torch.norm(s_rep - rand_noise, p=2) + torch.norm(v_rep - mu, p=2)
        elif self.loss_type == 'perturb_v_output_and_s_output':
            x_s, x_v, x_v_att = causal_model.split_x(inputs, labels, eval=True)
            v_rep = causal_model.get_conf_rep(x_v)
            v_pred = causal_model.get_conf_pred(v_rep)
            s_pred = causal_model(x_s)
            if new_label is not None:
                loss = F.cross_entropy(v_pred, new_label) + F.cross_entropy(s_pred, new_label)
            else:
                loss = (-1) * (F.cross_entropy(v_pred, labels) + F.cross_entropy(s_pred, new_label))
        else:
            raise ValueError(f"Unrecognized loss type {self.loss_type}")

        return self.beta * loss
