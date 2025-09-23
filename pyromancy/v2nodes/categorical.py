from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..params import eparameters
from .base import VariationalNode


@eparameters("logits")
class CategoricalNode(VariationalNode):
    r"""Categorical distribution parameterized by logits.

    Args:
        mask_probs (bool, optional): if zero values for input probabilities should
            be masked to ``-inf``. Defaults to False.
        eps_probs (float, optional): minimum values input probabilities will be
            clamped to. Defaults to 1e-12.

    Attributes:
        logits (~torch.nn.parameter.Parameter): current value of the node
            :math:`\log q`.

    Raises:
        ValueError: ``eps_probs`` must be nonnegative.

    Important:
        When ``mask_probs`` is True, then input probabilities are not bounded by
        ``eps_probs``. In practice, this will lead to numerical issues under certain
        conditions, such as when computing energy using KL-divergence.

    Tip:
        It is *strongly* recommended that probabilities are only passed in for
        manual initialization, and that logits are passed in elsewhere.
    """

    logits: nn.Parameter
    _mask_probs: bool
    _eps_probs: float

    def __init__(
        self,
        *shape: int | None,
        mask_probs: bool = True,
        eps_probs: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        VariationalNode.__init__(self, *shape, **kwargs)

        if eps_probs < 0:
            raise ValueError(f"`eps_probs` must be nonnegative, received {eps_probs}")

        self.probs = nn.Parameter(torch.empty(0), True)
        self._mask_probs = bool(mask_probs)
        self._eps_probs = float(eps_probs)

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        r"""Converts logits to probabilities.

        Args:
            logits (~torch.Tensor): unnormalized logits.

        Returns:
            torch.Tensor: softmax normalized probabilities.
        """
        logits, pragma = self.shapeobj.coalesce(self.logits)

        probs = F.softmax(logits, dim=1)
        probs = self.shapeobj.disperse(probs, pragma)

        return probs

    def _probs_to_logits(self, probs: torch.Tensor) -> torch.Tensor:
        r"""Converts probabilities to logits.

        Args:
            probs (~torch.Tensor): normalized probabilities.

        Returns:
            torch.Tensor: unnormalized logits.
        """
        if self._mask_probs:
            probs = probs.masked_fill(probs == 0, float("-inf"))
        else:
            probs = probs.clamp_min(self._eps_probs)

        logits = probs.log()

        return logits

    @torch.no_grad()
    def _cdf_uniform_sample(
        self, probs: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        r"""Samples from a categorical distribution by taking the CDF.

        Args:
            probs (~torch.Tensor): normalized probabilities.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            torch.Tensor: samples from the distribution.
        """
        probs, pragma = self.shapeobj.coalesce(probs)

        cdf = probs.cumsum(1)
        cdf[:, -1] = 1.0

        uniforms = cdf.new_empty(cdf.size(0), 1).uniform_(generator=generator)
        idx = torch.searchsorted(cdf, uniforms, right=True).squeeze(1)

        y = torch.zeros_like(probs).scatter_(1, idx, 1.0)
        y = self.shapeobj.disperse(y, pragma)

        return y

    def _gumbel_softmax_sample(
        self,
        logits: torch.Tensor,
        tau: float,
        discrete: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""Samples from a categorical distribution using the Gumbel—Softmax trick.

        Args:
            logits (~torch.Tensor): unnormalized logits.
            tau (float): softmax temperature.
            discrete (bool): if outputs should be discretized into one-hot vectors.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.

        Returns:
            torch.Tensor: samples from the distribution.
        """
        logits, pragma = self.shapeobj.coalesce(logits)

        gumbels = -torch.empty_like(logits).exponential_(generator=generator).log()
        gumbels = (logits + gumbels) / tau

        y = F.softmax(gumbels, dim=1)

        if discrete:
            idx = y.argmax(1, keepdim=True)
            yd = torch.zeros_like(logits).scatter_(1, idx, 1.0)
            y = yd - y.detach() + y

        y = self.shapeobj.disperse(y, pragma)

        return y

    def activity(self, as_logits: bool = False, **kwargs: Any) -> torch.Tensor:
        r"""Activity of the node.

        Args:
            as_logits (bool, optional): if the activity should be returned as logits
                rather than as probabilities. Defaults to False.

        Returns:
            ~torch.Tensor: activity of the node.
        """
        if self.logits.numel() == 0:
            return self.logits
        elif as_logits:
            return self.logits
        else:
            return self._logits_to_probs(self.logits)

    def energy(
        self,
        pred: torch.Tensor,
        from_logits: bool = True,
        fn: Literal["kld", "ce"] = "kld",
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes batchwise energy between predictions and the node's activity.

        .. math::
            \begin{aligned}
                \mathcal{E}_\text{KL} &= \sum_k q(k)
                \log \frac{q(k)}{p(k)} \\
                \mathcal{E}_\text{CE} &= -\sum_k q(k) \log p(k)
            \end{aligned}

        Args:
            pred (~torch.Tensor): prediction of the node's activity, :math:`p`
                or :math:`\log p`.
            from_logits (bool, optional): if the prediction should be interpreted
                as raw logits rather than as probabilities. Defaults to True.
            fn (Literal["kld", "ce"], optional): mode for computing the energy.
                Defaults to "kld".

        Returns:
            ~torch.Tensor: batchwise energy between the activity and the predictions.

        Raises:
            ValueError: invalid ``fn`` specified.

        Info:
            The `fn` parameter controls how energy is computed.

            - Kullback–Leibler Divergence ("kld"): the node's activity is treated as the
              as the distribution's mean, and reverse KL-divergence is taken.
            - Cross-Entropy ("ce"): the node's activity is treated as the
              as the distribution's mean, and cross-entropy is taken.

            Recall that KL-divergence and cross-entropy vary by the entropy of the left-hand
            distribution.

            .. math::
                \operatorname{D}_\text{KL}(q \parallel p) = H(q, p) - H(q)
        """
        q_logits, pragma = self.shapeobj.coalesce(self.logits)
        q_probs = F.softmax(q_logits, dim=1)

        match fn:
            case "kld":
                if from_logits:
                    p_logits, _ = self.shapeobj.coalesce(pred)
                    Hqp = torch.logsumexp(p_logits, dim=1) - (p_logits * q_probs).sum(1)
                else:
                    p_probs, _ = self.shapeobj.coalesce(pred)
                    Hqp = -torch.special.xlogy(q_probs, p_probs)

                Hq = torch.logsumexp(q_logits, dim=1) - (q_logits * q_probs).sum(1)
                E = Hqp - Hq

            case "ce":
                if from_logits:
                    p_logits, _ = self.shapeobj.coalesce(pred)
                    Hqp = torch.logsumexp(p_logits, dim=1) - (p_logits * q_probs).sum(1)
                else:
                    p_probs, _ = self.shapeobj.coalesce(pred)
                    Hqp = -torch.special.xlogy(q_probs, p_probs)

                E = Hqp

            case _:
                raise ValueError(f"invalid `fn` '{fn}' specified")

        E = self.shapeobj.disperse(E, pragma, "plate")
        return E.flatten(1).sum(1)

    def error(
        self,
        pred: torch.Tensor,
        from_logits: bool = True,
        fn: Literal["kld", "ce"] = "kld",
        **kwargs: Any,
    ) -> torch.Tensor:
        return torch.empty(0)

    @torch.no_grad()
    def initialize(
        self,
        pred: torch.Tensor,
        sample: bool = False,
        generator: torch.Generator | None = None,
        from_logits: bool = True,
        as_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Initializes the node's state.

        Args:
            pred (~torch.Tensor): prediction for the basis of initialization.
            sample (bool, optional): if the activity should be initialized using
                random sampling. Defaults to False.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.
            from_logits (bool, optional): if the prediction should be interpreted
                as raw logits rather than as probabilities. Defaults to True.
            as_logits (bool, optional): if the activity should be returned as logits
                rather than as probabilities. Defaults to False.
            procedure (Literal["cdf-uniform", "gumbel-softmax"], optional):
                sampling procedure to use when ``sample=True``. Defaults to "gumbel-softmax-continuous".
            temperature (float, optional): softmax temperature used by Gumbel–Softmax methods.
                Defaults to 1.0.
            discrete (bool, optional): if samples with Gumble–Softmax should be discretized
                into one-hot vectors. Defaults to False.
        """
        if sample:
            logits = self.sample(
                pred,
                generator=generator,
                from_logits=from_logits,
                as_logits=True,
                **kwargs,
            )
        else:
            logits = self.prediction(
                pred, from_logits=from_logits, as_logits=True, **kwargs
            )

        if not self.shapeobj.compat(*logits.shape):
            raise ValueError(
                f"shape specified by `pred` {(*logits.shape,)} "
                f"is incompatible with node shape {(*self.shapeobj,)}"
            )

        self.logits.data = self.logits.data.new_empty(*logits.shape)
        self.logits.copy_(logits)

        return self.activity(as_logits=as_logits, **kwargs)

    def prediction(
        self,
        pred: torch.Tensor,
        from_logits: bool = True,
        as_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Computes the single prediction for the node.

        Args:
            pred (~torch.Tensor): prediction for the basis of initialization.
            from_logits (bool, optional): if the prediction should be interpreted
                as raw logits rather than as probabilities. Defaults to True.
            as_logits (bool, optional): if the activity should be returned as logits
                rather than as probabilities. Defaults to False.

        Returns:
            ~torch.Tensor: prediction of the node's activity.

        Important:
            Every input must have the same shape.
        """
        if from_logits:
            if as_logits:
                return pred
            else:
                return self._logits_to_probs(pred)
        else:
            if as_logits:
                return self._probs_to_logits(pred)
            else:
                return pred

    @torch.no_grad()
    def reset(self, **kwargs: Any) -> None:
        r"""Resets the node state."""
        self.zero_grad()
        self.logits.data = self.logits.new_empty(0)

    def sample(
        self,
        pred: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        from_logits: bool = True,
        as_logits: bool = False,
        procedure: Literal["cdf-uniform", "gumbel-softmax"] = "gumbel-softmax",
        temperature: float = 1.0,
        discrete: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Samples from the learned conditional distribution.

        Args:
            pred (~torch.Tensor | None, optional): prediction of the node's activity,
                used as the parameters of the distribution if provided. If None, the
                node's activity is used for the prediction. Defaults to None.
            generator (~torch.Generator | None, optional): pseudorandom number generator
                for sampling. Defaults to None.
            from_logits (bool | None, optional): if the prediction should be interpreted
                as raw logits rather than as probabilities. Defaults to True.
            as_logits (bool, optional): if the activity should be returned as logits
                rather than as probabilities. Defaults to False.
            procedure (Literal["cdf-uniform", "gumbel-softmax"], optional):
                sampling procedure to use. Defaults to "gumbel-softmax".
            temperature (float, optional): softmax temperature used by Gumbel–Softmax.
                Defaults to 1.0.
            discrete (bool, optional): if samples with Gumble–Softmax should be discretized
                into one-hot vectors. Defaults to False.

        Returns:
            ~torch.Tensor: samples from the conditional distribution.

        Raises:
            ValueError: invalid ``procedure`` specified.

        Caution:
            The sampling procedure ``"cdf-uniform"`` is non-differentiable.

        Note:
            If ``pred`` is not specified, then ``from_logits`` is ignored and the
            interal logits will be used.
        """
        if pred is None:
            pred = self.logits
            from_logits = True

        match procedure:
            case "cdf-uniform":
                if from_logits:
                    pred = self._logits_to_probs(pred)
                z = self._cdf_uniform_sample(pred, generator=generator)
            case "gumbel-softmax":
                if not from_logits:
                    pred = self._probs_to_logits(pred)
                z = self._gumbel_softmax_sample(
                    pred, temperature, discrete, generator=generator
                )
            case _:
                raise ValueError(f"invalid `procedure` '{procedure}' specified")

        if as_logits:
            z = self._probs_to_logits(z)

        return z
