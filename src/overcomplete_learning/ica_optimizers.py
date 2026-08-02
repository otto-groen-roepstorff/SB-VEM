from __future__ import division
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

__authors__ = "Jesse Livezey, Alex Bujan"
__ported_by__ = "Theano -> PyTorch port for Python 3.11"


# ---------------------------------------------------------------------------
# Helper: normalise rows of W
# ---------------------------------------------------------------------------

def _normalize_rows(W: torch.Tensor) -> torch.Tensor:
    epssumsq = (W ** 2).sum(dim=1, keepdim=True).clamp(min=1e-7)
    return W / torch.sqrt(epssumsq)


# ---------------------------------------------------------------------------
# Learning-rule update functions
# (kept for API compatibility; prefer torch.optim in new code)
# ---------------------------------------------------------------------------

def sgd(params, grads, learning_rate=0.01):
    updates = {}
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates


def momentum(params, grads, learning_rate=0.01, momentum=0.1, nesterov=True):
    """Nesterov momentum."""
    updates = {}
    velocities = {}
    for param, grad in zip(params, grads):
        v = torch.zeros_like(param)
        v_new = v * momentum - learning_rate * grad
        delta = v_new
        if nesterov:
            delta = momentum * v_new - learning_rate * grad
        updates[param] = param + delta
        velocities[param] = v_new
    return updates, velocities


def adam(params, grads, learning_rate=0.001, beta1=0.9, beta2=0.999,
         epsilon=1e-8):
    updates = {}
    t = torch.tensor(1.0)
    a_t = learning_rate * torch.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

    for param, grad in zip(params, grads):
        m_prev = torch.zeros_like(param)
        v_prev = torch.zeros_like(param)

        m_t = beta1 * m_prev + (1.0 - beta1) * grad
        v_t = beta2 * v_prev + (1.0 - beta2) * grad ** 2
        step = a_t * m_t / (torch.sqrt(v_t) + epsilon)
        updates[param] = param - step
    return updates


# ---------------------------------------------------------------------------
# Base Optimizer
# ---------------------------------------------------------------------------

class Optimizer:
    """Base optimiser — subclasses implement fit()."""

    def __init__(self, prior='soft', verbose=False, **fit_kwargs):
        self.verbose = verbose
        self.prior = prior
        self.fit_kwargs = fit_kwargs
        self.setup(**fit_kwargs)

    # ------------------------------------------------------------------
    # Public transform / reconstruct helpers
    # ------------------------------------------------------------------

    def _transforms(self, W: torch.Tensor, X: torch.Tensor):
        S = W @ X
        X_hat = W.T @ S
        return S, X_hat

    def transform(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        Xt = torch.tensor(X, dtype=torch.float32)
        Wt = torch.tensor(W, dtype=torch.float32)
        Wn = _normalize_rows(Wt)
        S, _ = self._transforms(Wn, Xt)
        return S.detach().numpy()

    def reconstruct(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        Xt = torch.tensor(X, dtype=torch.float32)
        Wt = torch.tensor(W, dtype=torch.float32)
        Wn = _normalize_rows(Wt)
        _, X_hat = self._transforms(Wn, Xt)
        return X_hat.detach().numpy()

    def losses(self, X: np.ndarray, W: np.ndarray):
        Xt = torch.tensor(X, dtype=torch.float32)
        Wt = torch.tensor(W, dtype=torch.float32)
        norm_projection = self.fit_kwargs.get('norm_projection', True)
        Wn = _normalize_rows(Wt) if norm_projection else Wt
        loss, error, penalty, mse, _, _ = self.cost(Wn, Xt, **self.fit_kwargs)
        to_np = lambda t: t.item() if isinstance(t, torch.Tensor) else float(t)
        return [to_np(v) for v in (loss, error, penalty, mse)]

    # ------------------------------------------------------------------
    # Cost function (pure PyTorch, no compilation needed)
    # ------------------------------------------------------------------

    def cost(self, Wn: torch.Tensor, X: torch.Tensor,
             degeneracy=None, lambd=0., p=None, **kwargs):

        L, D = Wn.shape
        minC = torch.sqrt(
            torch.tensor((L - D) / (float(D) * (L - 1)), dtype=torch.float32)
        )

        S, X_hat = self._transforms(Wn, X)
        gram = Wn @ Wn.T
        gram_diff = gram - torch.eye(gram.shape[0])

        loss = None
        assert lambd >= 0.

        # Normalise Lp shorthand
        if degeneracy == 'L2':
            degeneracy, p = 'Lp', 2
        elif degeneracy == 'L4':
            degeneracy, p = 'Lp', 4

        if isinstance(degeneracy, str) and degeneracy and degeneracy[0] == 'L':
            try:
                p = int(degeneracy[1:])
                degeneracy = 'Lp'
            except ValueError:
                pass

        # ---- error term --------------------------------------------------
        if degeneracy == 'RICA':
            error = 0.5 * ((X_hat - X) ** 2).sum(dim=0).mean()

        elif degeneracy == 'M1':
            sign = torch.sign(gram).detach()
            sign[sign == 0] = 1.
            target = sign * ((1. - minC) * torch.eye(L) + minC)
            error = ((gram - target) ** 2).sum()

        elif degeneracy == 'M2':
            error = ((gram ** 2 - ((1. - minC ** 2) * torch.eye(L) +
                      minC ** 2)) ** 2).sum()

        elif degeneracy == 'Lp':
            assert isinstance(p, int)
            if p % 2 == 0:
                err = gram_diff
                for _ in range(p // 2):
                    err = err ** 2
            else:
                err = torch.ones_like(gram_diff)
                if p > 1:
                    err = gram_diff
                    for _ in range(p // 2):
                        err = err ** 2
                err = err * gram_diff.abs()
            error = (1. / p) * err.sum()

        elif degeneracy == 'COULOMB':
            epsilon = 0.01
            error = (1. / torch.sqrt(1. + epsilon - gram_diff ** 2) -
                     1. / np.sqrt(1. + epsilon)).sum()

        elif degeneracy == 'COULOMB_F':
            epsilon = 0.01
            error = (1. / torch.sqrt(1. + epsilon - gram_diff ** 2)
                     - 0.5 * gram_diff ** 2 / (1. + epsilon) ** 1.5
                     - 1. / np.sqrt(1. + epsilon)).sum()

        elif degeneracy == 'RANDOM':
            epsilon = 0.01
            error = -(torch.log(1. + epsilon - gram_diff ** 2)
                      - np.log(1. + epsilon)).sum()

        elif degeneracy == 'RANDOM_F':
            epsilon = 0.01
            error = -(torch.log(1. + epsilon - gram_diff ** 2)
                      - np.log(1. + epsilon)
                      + gram_diff ** 2 / (1. + epsilon)).sum()

        elif degeneracy == 'COHERENCE_SOFT':
            agd = gram_diff.abs()
            agds = agd ** 2
            gs = gram ** 2
            boundary = gs.mean().detach()
            error = torch.where(agds > boundary, agds,
                                torch.zeros_like(agds)).sum()

        elif degeneracy == 'COHERENCE':
            error = gram_diff.abs().max()

        elif degeneracy == 'SM':
            ts = torch.tanh(S)
            score = -(Wn.T @ ts)
            dscore = -((Wn ** 2).T @ (1. - ts ** 2))
            error = (dscore + 0.5 * score ** 2).sum(dim=0).mean()

        elif degeneracy is None:
            error = None

        else:
            raise ValueError(f"Unknown degeneracy: {degeneracy}")

        # ---- loss assembly -----------------------------------------------
        if degeneracy is not None and not np.isinf(lambd):
            loss = error

        if np.isinf(lambd) or degeneracy is None:
            penalty = (torch.log(torch.cosh(S)).sum(dim=0).mean()
                       if self.prior == 'soft'
                       else S.abs().sum(dim=0).mean())
            loss = penalty
        elif degeneracy == 'SM':
            assert lambd == 0.
            penalty = torch.log(torch.cosh(S)).sum(dim=0).mean()
        elif lambd > 0.:
            penalty = (torch.log(torch.cosh(S)).sum(dim=0).mean()
                       if self.prior == 'soft'
                       else S.abs().sum(dim=0).mean())
            loss = loss + lambd * penalty
        else:
            penalty = None

        # ---- MSE on normalised vectors -----------------------------------
        eps = 1e-8
        X_normed = X / (X ** 2).sum(dim=0, keepdim=True).sqrt().clamp(min=eps)
        X_hat_normed = (X_hat /
                        (X_hat ** 2).sum(dim=0, keepdim=True).sqrt().clamp(min=eps))
        mse = ((X_normed - X_hat_normed) ** 2).sum(dim=0).mean()

        return loss, error, penalty, mse, S, X_hat

    def fit(self, data, components_):
        raise NotImplementedError

    def setup(self, **kwargs):
        pass


# ---------------------------------------------------------------------------
# L-BFGS-B Optimizer
# ---------------------------------------------------------------------------

class LBFGSB(Optimizer):

    def setup(self, n_sources, n_mixtures, degeneracy, lambd, p,
              **kwargs):
        self.n_sources = n_sources
        self.n_mixtures = n_mixtures
        self.degeneracy = degeneracy
        self.lambd = lambd
        self.p = p
        self.norm_projection = kwargs.get('norm_projection', True)

    def _f_df(self, w_np: np.ndarray, X: torch.Tensor):
        extra = {k: v for k, v in self.fit_kwargs.items()
                 if k not in ('degeneracy', 'lambd', 'p')}
        Wv = torch.tensor(w_np, dtype=torch.float32, requires_grad=True)
        W = Wv.reshape(self.n_sources, self.n_mixtures)
        Wn = _normalize_rows(W) if self.norm_projection else W
        loss, _, _, _, _, _ = self.cost(Wn, X,
                                        self.degeneracy, self.lambd, self.p,
                                        **extra)
        loss.backward()
        grad = Wv.grad.detach().numpy().astype(np.float64)
        return loss.item(), grad

    def fit(self, data: np.ndarray, components_: np.ndarray) -> np.ndarray:
        X = torch.tensor(data, dtype=torch.float32)

        def float_f_df(w):
            loss_val, grad_val = self._f_df(w, X)
            return float(loss_val), grad_val

        def callback(w):
            extra = {k: v for k, v in self.fit_kwargs.items()
                     if k not in ('degeneracy', 'lambd', 'p')}
            Wv = torch.tensor(w, dtype=torch.float32)
            W = Wv.reshape(self.n_sources, self.n_mixtures)
            Wn = _normalize_rows(W) if self.norm_projection else W
            loss, error, penalty, mse, _, _ = self.cost(
                Wn, X, self.degeneracy, self.lambd, self.p, **extra)        
            to_f = lambda t: t.item() if isinstance(t, torch.Tensor) else 0.
            print('Loss: {:.6f}, Error: {:.6f}, Penalty: {:.6f}, MSE: {:.6f}'.format(
                to_f(loss), to_f(error), to_f(penalty) if penalty is not None else 0.,
                to_f(mse)))

        cb = callback if self.verbose else None
        w0 = components_.ravel().astype(np.float64)
        res = minimize(float_f_df, w0, jac=True, method='L-BFGS-B', callback=cb)
        w_f = res.x
        print('ICA with L-BFGS-B done!')
        print('Final loss value: {:.6f}'.format(res.fun))
        return w_f.reshape(components_.shape)


# ---------------------------------------------------------------------------
# SGD Optimizer
# ---------------------------------------------------------------------------

class SGD(Optimizer):

    def setup(self, n_sources, n_mixtures, w_0, lambd, degeneracy,
              learning_rule, p, **kwargs):
        self.n_sources = n_sources
        self.n_mixtures = n_mixtures
        self.lambd = lambd
        self.degeneracy = degeneracy
        self.learning_rule = learning_rule
        self.p = p
        self.norm_projection = kwargs.get('norm_projection', True)

        # Initialise W as a torch parameter
        if w_0 is not None:
            init = torch.tensor(w_0, dtype=torch.float32)
        else:
            init = torch.randn(n_sources, n_mixtures, dtype=torch.float32)
        self.W = torch.nn.Parameter(init)

        # Map learning_rule string/callable to torch.optim
        lr = kwargs.get('learning_rate', 0.01)
        mom = kwargs.get('momentum_val', 0.1)
        if learning_rule is momentum or learning_rule == 'momentum':
            self.optimizer = torch.optim.SGD(
                [self.W], lr=lr, momentum=mom, nesterov=True)
        elif learning_rule is adam or learning_rule == 'adam':
            self.optimizer = torch.optim.Adam([self.W], lr=lr)
        else:
            # Default: plain SGD
            self.optimizer = torch.optim.SGD([self.W], lr=lr)

    def _train_step(self, batch: torch.Tensor):
        extra = {k: v for k, v in self.fit_kwargs.items()
             if k not in ('degeneracy', 'lambd', 'p')}
        self.optimizer.zero_grad()
        Wn = _normalize_rows(self.W) if self.norm_projection else self.W
        loss, error, penalty, mse, _, _ = self.cost(
            Wn, batch, self.degeneracy, self.lambd, self.p, **extra)
        loss.backward()
        self.optimizer.step()
        to_f = lambda t: t.item() if isinstance(t, torch.Tensor) else 0.
        return (to_f(loss),
                to_f(error) if error is not None else 0.,
                to_f(penalty) if penalty is not None else 0.,
                to_f(mse))

    def fit(self, data: np.ndarray, components_: np.ndarray,
            tol=1e-4, batch_size=128, n_epochs=1000000,
            patience=10000, seed=20160615) -> np.ndarray:

        n_examples = data.shape[1]
        rng = np.random.RandomState(seed)
        self.W.data = torch.tensor(components_, dtype=torch.float32)

        n_batches = -(-n_examples // batch_size)   # ceiling division
        lowest_cost = np.inf
        improve = 0

        for ii in range(n_epochs):
            order = rng.permutation(n_examples)
            cur_cost = error = penalty = mse = 0.

            for jj in range(n_batches):
                start = jj * batch_size
                end = (jj + 1) * batch_size
                batch = torch.tensor(
                    data[:, order[start:end]], dtype=torch.float32)
                res = self._train_step(batch)
                n = batch.shape[1]
                cur_cost += res[0] * n
                error    += res[1] * n
                penalty  += res[2] * n
                mse      += res[3] * n

            cur_cost /= n_examples
            error    /= n_examples
            penalty  /= n_examples
            mse      /= n_examples

            if self.verbose:
                print('Loss: {:.6f}, Error: {:.6f}, '
                      'Penalty: {:.6f}, MSE: {:.6f}'.format(
                          cur_cost, error, penalty, mse))

            if cur_cost < lowest_cost - tol:
                lowest_cost = cur_cost
                improve = 0
            else:
                improve += 1
            if improve == patience:
                break

        print('ICA with SGD done! Epoch {}'.format(ii))
        print('Final loss value: {:.6f}'.format(cur_cost))
        return self.W.detach().numpy()
