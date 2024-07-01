# Inherited from https://github.com/HazyResearch/KGEmb
"""Knowledge Graph embedding model optimizer."""
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from datasets.kg_dataset import KGDataset3
from models.gnnbase import GNN
from typing import Optional
import time


class KGOptimizer(object):
    """Knowledge Graph embedding model optimizer.

    KGOptimizers performs loss computations with negative sampling and gradient descent steps.

    Attributes:
        model: models.base.KGModel
        regularizer: regularizers.Regularizer
        optimizer: torch.optim.Optimizer
        batch_size: An integer for the training batch size
        neg_sample_size: An integer for the number of negative samples
        double_neg: A boolean (True to sample both head and tail entities)
    """

    def __init__(
            self, model, regularizer, optimizer, batch_size,
            update_steps,
            neg_sample_size, double_neg, optimizer2=None,
            loss = "crossentropy",
            smoothing=None,
            verbose=True,):
        """Inits KGOptimizer."""
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.optimizer.zero_grad()
        self.optimizer2 = optimizer2
        if self.optimizer2 is not None:
            self.optimizer2.zero_grad()
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.verbose = verbose
        self.double_neg = double_neg
        self.ce = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0 if smoothing is None else smoothing)
        self.bce = nn.BCELoss(reduction='mean')
        self.neg_sample_size = neg_sample_size
        self.n_entities = model.sizes[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.smoothing = smoothing
        self.loss = loss
        

    def reduce_lr(self, factor=0.8):
        """Reduce learning rate.

        Args:
            factor: float for the learning rate decay
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        # if self.optimizer2 is not None:
        #     for param_group in self.optimizer2.param_groups:
        #         param_group['lr'] *= factor

    def get_neg_samples(self, input_batch):
        """Sample negative examples.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            negative_batch: torch.Tensor of shape (neg_sample_size x 3) with negative examples
        """
        # negative_batch = input_batch.repeat(self.neg_sample_size, 1)
        # batch_size = input_batch.shape[0]
        # negsamples = torch.Tensor(np.random.randint(
        #     self.n_entities,
        #     size=batch_size * self.neg_sample_size)
        # ).to(input_batch.dtype)
        # negative_batch[:, 2] = negsamples
        # if self.double_neg:
        #     negsamples = torch.Tensor(np.random.randint(
        #         self.n_entities,
        #         size=batch_size * self.neg_sample_size)
        #     ).to(input_batch.dtype)
        #     negative_batch[:, 0] = negsamples
        # return negative_batch
        negsamples = torch.randint(
            0, self.n_entities - 1,
            size=(input_batch.shape[0], self.neg_sample_size),
            device=input_batch.device
        )
        negsamples = torch.where(negsamples < input_batch[:, 2].unsqueeze(-1), negsamples, negsamples + 1)

        return negsamples

    def neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss with negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.

        Returns:
            loss: torch.Tensor with negative sampling embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        # positive samples
        if isinstance(input_batch, tuple):
            input_batch, labels = input_batch

        positive_score, factors = self.model(input_batch[:, :2].unsqueeze(1), input_batch[:, 2].unsqueeze(1))
        positive_score = F.logsigmoid(positive_score)

        # negative samples 
        neg_samples = self.get_neg_samples(input_batch)
        negative_score, _ = self.model(input_batch[:, :2].unsqueeze(1), neg_samples)
        negative_score = F.logsigmoid(-negative_score)
        loss = - torch.cat([positive_score.view(-1), negative_score.view(-1)]).mean()
        return loss, factors

    # def no_neg_sampling_loss(self, input_batch):
    #     """Compute KG embedding loss without negative sampling.

    #     Args:
    #         input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

    #     Returns:
    #         loss: torch.Tensor with embedding loss
    #         factors: torch.Tensor with embeddings weights to regularize
    #     """
    #     predictions, factors = self.model(input_batch)
    #     truth = input_batch[:, 2]
    #     log_prob = F.logsigmoid(-predictions)
    #     idx = torch.arange(0, truth.shape[0], dtype=truth.dtype)
    #     pos_scores = F.logsigmoid(predictions[idx, truth]) - F.logsigmoid(-predictions[idx, truth])
    #     log_prob[idx, truth] += pos_scores
    #     loss = - log_prob.mean()
    #     loss += self.regularizer.forward(factors)
    #     return loss, factors

    def no_neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss without negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        if isinstance(input_batch, tuple):
            input_batch, labels = input_batch
            # If smoothing is not None, we use label smoothing
            predictions, factors = self.model(input_batch)
            if not self.smoothing is None:
                labels = (1.0 - self.smoothing) * labels.to(predictions.dtype) + self.smoothing / self.n_entities
            loss = self.bce(predictions.sigmoid(), labels)
            loss += self.regularizer.forward(factors)
        else:
            predictions, factors = self.model(input_batch)
            truth = input_batch[:, 2]
            log_prob = F.logsigmoid(-predictions)
            idx = torch.arange(0, truth.shape[0], dtype=truth.dtype)
            pos_scores = F.logsigmoid(predictions[idx, truth]) - F.logsigmoid(-predictions[idx, truth])
            log_prob[idx, truth] += pos_scores
            loss = - log_prob.mean()
            loss += self.regularizer.forward(factors)
        return loss, factors

    def calculate_loss(self, input_batch):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss and regularization loss
        """
        if self.neg_sample_size > 0:
            loss, factors = self.neg_sampling_loss(input_batch)
        else:
            if self.loss == "crossentropy":
                if isinstance(input_batch, tuple):
                    input_batch = input_batch[0]
                predictions, factors = self.model(input_batch)
                truth = input_batch[:, 2]
                loss = self.ce(predictions, truth.unsqueeze(1))
            elif self.loss == "binarycrossentropy":
                loss, factors = self.no_neg_sampling_loss(input_batch)

        # regularization loss
        loss += self.regularizer.forward(factors)
        return loss

    def calculate_valid_loss(self, examples):
        """Compute KG embedding loss over validation examples.

        Args:
            examples: torch.LongTensor of shape (N_valid x 3) with validation triples

        Returns:
            loss: torch.Tensor with loss averaged over all validation examples
        """
        if not isinstance(examples, tuple):
            if isinstance(examples, np.ndarray):
                examples = torch.from_numpy(examples)
            b_begin = 0
            loss = 0.0
            counter = 0
            with torch.no_grad():
                while b_begin < examples.shape[0]:
                    input_batch = examples[b_begin:b_begin + self.batch_size].to(self.device)
                    b_begin += self.batch_size
                    loss += self.calculate_loss(input_batch)
                    counter += 1
            loss /= counter
        else:
            examples_valid, labels = examples
            if isinstance(examples_valid, np.ndarray):
                examples_valid = torch.from_numpy(examples_valid)
            b_begin = 0
            loss = 0.0
            counter = 0
            with torch.no_grad():
                while b_begin < examples_valid.shape[0]:
                    input_batch = examples_valid[b_begin:b_begin + self.batch_size].to(self.device)
                    input_label = labels[b_begin:b_begin + self.batch_size].toarray()
                    input_label = torch.from_numpy(input_label).to(self.device).unsqueeze(-1)
                    b_begin += self.batch_size
                    loss += self.calculate_loss((input_batch, input_label))
                    counter += 1
            loss /= counter
        return loss

    def epoch(self, examples):
        """Runs one epoch of training KG embedding model.

        Args:
            examples: torch.LongTensor of shape (N_train x 3) with training triples

        Returns:
            loss: torch.Tensor with loss averaged over all training examples
        """
        if not isinstance(examples, tuple):
            actual_examples = examples[torch.randperm(examples.shape[0]), :]
            with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss')
                b_begin = 0
                total_loss = 0.0
                counter = 0
                while b_begin < examples.shape[0]:
                    input_batch = actual_examples[b_begin:b_begin + self.batch_size].to(self.device)

                    # gradient step
                    l = self.calculate_loss(input_batch)
                    l.backward()

                    if self.update_steps == 1 or \
                        (counter + 1) % self.update_steps == 0 or \
                            b_begin + self.batch_size >= examples.shape[0]:
                        self.optimizer.step()
                        if self.optimizer2 is not None:
                            self.optimizer2.step()
                        self.optimizer.zero_grad()
                        if self.optimizer2 is not None:
                            self.optimizer2.zero_grad()

                    b_begin += self.batch_size
                    total_loss += l.item()
                    counter += 1
                    bar.update(input_batch.shape[0])
                    bar.set_postfix(loss=f'{l.item():.4f}')
            total_loss /= counter
        else:
            examples, labels = examples
            perm = np.random.permutation(len(examples))
            if isinstance(examples, torch.Tensor):
                actual_examples = examples[torch.from_numpy(perm), :]
            else:
                actual_examples = torch.from_numpy(examples[perm])
            actual_labels = labels[perm]
            with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f'train loss')
                b_begin = 0
                total_loss = 0.0
                counter = 0
                while b_begin < examples.shape[0]:
                    input_batch = actual_examples[b_begin:b_begin + self.batch_size].to(self.device)
                    input_label = actual_labels[b_begin:b_begin + self.batch_size].toarray()
                    input_label = torch.from_numpy(input_label).to(self.device).unsqueeze(-1)

                    # gradient step
                    l = self.calculate_loss((input_batch, input_label))
                    l.backward()

                    if self.update_steps == 1 or \
                        (counter + 1) % self.update_steps == 0 or \
                            b_begin + self.batch_size >= examples.shape[0]:
                        self.optimizer.step()
                        if self.optimizer2 is not None:
                            self.optimizer2.step()
                        self.optimizer.zero_grad()
                        if self.optimizer2 is not None:
                            self.optimizer2.zero_grad()

                    b_begin += self.batch_size
                    total_loss += l.item()
                    counter += 1
                    bar.update(input_batch.shape[0])
                    bar.set_postfix(loss=f'{l.item():.4f}')
            total_loss /= counter
        return total_loss


class KGOptimizerSubgraph(KGOptimizer):
    """Knowledge Graph embedding model optimizer with subgraph sampling.

    KGOptimizerSubgraph
    """
    def __init__(self, model, regularizer, optimizer, batch_size, update_steps, neg_sample_size, double_neg, optimizer2=None, loss="crossentropy", smoothing=None, verbose=True, dataset: Optional[KGDataset3] = None):
        assert isinstance(model, GNN), f"Model {type(model)} must be a GNN"
        super(KGOptimizerSubgraph, self).__init__(model=model, regularizer=regularizer, optimizer=optimizer, batch_size=batch_size,
                                                  update_steps=update_steps, neg_sample_size=neg_sample_size, double_neg=double_neg,
                                                    optimizer2=optimizer2, loss=loss, smoothing=smoothing, verbose=verbose)
        self.dataset = dataset
        self.loader = self.dataset.make_loader(batch_size=self.batch_size, shuffle=True, num_workers=4)

    def epoch(self, examples):
        """Runs one epoch of training KG embedding model.

        Args:
            examples: torch.LongTensor of shape (N_train x 3) with training triples

        Returns:
            loss: torch.Tensor with loss averaged over all training examples
        """
        with tqdm.tqdm(self.loader, unit='batch', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            total_loss = 0.0
            counter = 0
            for batch in bar:
                l = self.calculate_loss(batch.to(self.dataset.g_device), optimizer=self.optimizer)
                total_loss += l
                counter += 1
                bar.set_postfix(loss=f'{l:.4f}')
            total_loss /= counter
        return total_loss
    
    def calculate_loss(self, input_batch, split="train", optimizer=None):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: tuple containing a Data object with the subgraph and a torch sparse tensor with labels
        Returns:
            loss: torch.Tensor with embedding loss and regularization loss
        """
        if self.neg_sample_size > 0:
            loss, factors = self.neg_sampling_loss(input_batch)
        else:
            subgraph, queries, labels = self.dataset.make_subgraph(input_batch, split=split, return_labels=True)
            # Make triples
            subgraph = subgraph.to(self.device)

            # Here, we can process the subgraph by minibatch
            batch_size = self.batch_size
            b_begin = 0
            total_subgraph_loss = 0.0
            counter = 0

            while b_begin < queries.shape[0]:
                queries_batch = queries[b_begin:b_begin + batch_size].to(self.device)
                labels_batch = torch.index_select(
                    labels,
                    0,
                    torch.arange(start=b_begin, end=min(b_begin + batch_size, queries.shape[0]), device=labels.device)
                )
                predictions, factors = self.model(
                    queries = queries_batch, tails=None,
                    x=subgraph.x,
                    edge_index=subgraph.edge_index[:, subgraph.train_mask],
                    edge_type=subgraph.edge_type[subgraph.train_mask],
                )
                if self.loss == "crossentropy":
                    truth = queries_batch[:, 2]
                    loss = self.ce(predictions, truth.unsqueeze(1))
                elif self.loss == "binarycrossentropy":
                    labels_batch = labels_batch.to(self.device).to_dense().unsqueeze(-1)
                    labels_batch = (1.0 - self.smoothing) * labels_batch.to(predictions.dtype) + self.smoothing / subgraph.num_nodes
                    loss = self.bce(predictions.sigmoid(), labels_batch)
                loss += self.regularizer.forward(factors)
                if optimizer is not None:
                    loss.backward()
                    if self.update_steps == 1 or \
                        (counter + 1) % self.update_steps == 0 or \
                            b_begin + self.batch_size >= queries.shape[0]:
                        self.optimizer.step()
                        if self.optimizer2 is not None:
                            self.optimizer2.step()
                        self.optimizer.zero_grad()
                        if self.optimizer2 is not None:
                            self.optimizer2.zero_grad()
                
                b_begin += batch_size
                counter += 1
                total_subgraph_loss += loss.item()
        # regularization loss
        return total_subgraph_loss / counter
    
    def calculate_valid_loss(self, examples):
        labels = self.dataset.make_labels(
            self.dataset.g,
            split="train",
            triples=examples
        )
        with torch.no_grad():
            b_begin = 0
            loss = 0.0
            counter = 0
            while b_begin < examples.shape[0]:
                input_batch = examples[b_begin:b_begin + self.batch_size].to(self.device)
                b_top = min(b_begin + self.batch_size, examples.shape[0])
                r = torch.arange(start=b_begin, end=b_top, device=labels.device)
                input_label = torch.index_select(labels, 0, r).to(self.device)
                b_begin += self.batch_size

                predictions, factors = self.model(
                    queries = input_batch, tails=None
                )
                if self.loss == "crossentropy":
                    truth = input_batch[:, 2]
                    loss += self.ce(predictions, truth.unsqueeze(1))
                elif self.loss == "binarycrossentropy":
                    input_label = input_label.to_dense().unsqueeze(-1)
                    input_label = (1.0 - self.smoothing) * input_label.to(predictions.dtype) + self.smoothing / self.dataset.n_entities
                    loss += self.bce(predictions.sigmoid(), input_label)

                counter += 1
                loss += self.regularizer.forward(factors)
        loss /= counter
        return loss




