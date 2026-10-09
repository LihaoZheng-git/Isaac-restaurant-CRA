"""Independent, offline numeric inference for the audited 2017 Image2Mass files.

This is an evaluation bridge, not a robot runtime backend. It interprets a
restricted layer graph using existing PyTorch and OpenCV installations. Legacy
Lambda byte strings are hash checked and mapped to reviewed arithmetic; they
are never executed. Equivalence needs published-prediction checks (see report).
Upstream source, weights, and numeric tables are not distributed with this file.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


METRES_PER_INCH = 0.0254
KG_PER_POUND = 0.45359237
_LAMBDA_OPERATIONS = {
    "ca6269a31991332a0305b825318fde7ca7c960bcaee035c5ab4d45381f589816": "rgb",
    "313d6f43859c5d5afbc80cae448057e7cbcb927f59e889c91baa5a12345e0098": "rgb",
    "d43c749284765591cf6d44545071bf66f386ef276327808535d24fcd8b266652": "triple",
    "54d621c69f3f0f002207673818c18a5181ca144d940739b3f40ae9cfac98c6ac": "exlu",
    "521ac5de3355ff67dc6d55725c86509cb1808a20282789d8fa8915dcd3adaef5": "density",
}


def d4(image: np.ndarray, index: int, *, inverse: bool = False) -> np.ndarray:
    """Eight square image transforms; inverse reverses the operation order."""
    if index not in range(8):
        raise ValueError("D4 index must be in [0, 7]")
    result = image
    if index >= 4 and not inverse:
        result = np.rot90(result)
    flip = index % 4 if index < 4 else (0, 2, 1, 3)[index % 4]
    if flip in (1, 3):
        result = np.fliplr(result)
    if flip in (2, 3):
        result = np.flipud(result)
    if index >= 4 and inverse:
        result = np.rot90(result, -1)
    return np.ascontiguousarray(result)


def legacy_rectangle(rect: tuple) -> tuple:
    """Restore the release's axis-aligned tie convention (0, swapped sides).

    OpenCV 4.13 returns -90 for these tied rectangles; the published release
    predictions use the geometrically identical 0-degree representation.
    Geometry is unchanged, but side order/angle are learned input features.
    """
    center, size, angle = rect
    if angle == -90.0:
        return center, size[::-1], 0.0
    return rect


def auxiliary_features(
    dimensions_inches: np.ndarray, thickness: np.ndarray, rect: tuple
) -> np.ndarray:
    """35 released geometry features, with explicit degenerate-input rejection."""
    dims = np.asarray(dimensions_inches, np.float64)
    if dims.shape != (3,) or not np.isfinite(dims).all() or (dims <= 0).any():
        raise ValueError("three positive dimensions in inches are required")
    if thickness.shape != (19, 19) or not np.isfinite(thickness).all():
        raise ValueError("finite 19x19 thickness is required")
    center, size, angle = rect
    rw, rh = size
    extents = np.asarray([rw, rh, float(thickness.max()) * 32])
    if rw <= 0 or rh <= 0 or extents[2] <= 0:
        raise ValueError("no usable foreground/thickness geometry")
    volume = float(np.prod(dims))
    envelope = float(np.prod(extents))
    area = rw * rh
    filled = float(thickness.sum()) / envelope
    visible = float(np.count_nonzero(thickness > 1 / 32))
    filled_2d = visible / area
    ordered = np.sort(dims)[::-1]
    a, b, c = ordered
    result = np.asarray(
        [
            volume,
            *dims,
            *extents,
            *center,
            angle,
            float(thickness.sum()) * 32,
            visible,
            area,
            envelope,
            filled,
            filled_2d,
            volume * filled,
            volume * filled_2d,
            volume * math.sqrt(filled),
            volume * math.sqrt(filled_2d),
            filled**2,
            filled_2d**2,
            *ordered,
            *(ordered**2),
            a * b,
            a * c,
            b * c,
            b / a,
            c / a,
            extents[1] / extents[0],
            extents[2] / extents[0],
        ],
        np.float32,
    )
    if result.shape != (35,) or not np.isfinite(result).all():
        raise ValueError("invalid auxiliary geometry features")
    return result


def tf_same_pad(
    value: Tensor,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
    *,
    fill: float = 0,
) -> Tensor:
    """TensorFlow SAME padding for NCHW, including asymmetric stride padding."""
    pads = []
    for length, k, s, d in zip(value.shape[-2:], kernel, stride, dilation):
        total = max(0, (math.ceil(length / s) - 1) * s + (k - 1) * d + 1 - length)
        pads.append((total // 2, total - total // 2))
    return F.pad(value, (*pads[1], *pads[0]), value=fill)


class NumericGraph:
    """Restricted channels-last inference graph for these two known checkpoints."""

    def __init__(self, folder: Path, name: str, device: str = "cpu") -> None:
        self.config = json.loads((folder / f"{name}.json").read_text("utf-8"))["config"]
        self.device = device
        with np.load(folder / f"{name}.npz", allow_pickle=False) as archive:
            self.weights = {
                key: torch.as_tensor(archive[key], device=device)
                for key in archive.files
            }
        self.density = torch.as_tensor(
            np.load(folder / "density.npy", allow_pickle=False), device=device
        )
        if self.density.ndim != 1 or len(self.density) < 2:
            raise ValueError("invalid density table")
        self.cache = {}

    def weight(self, prefix: str, layer: str, parameter: str) -> Tensor:
        key = (prefix, layer, parameter)
        if key not in self.cache:
            # The thickness release suffixes some nested weight names (_11),
            # while its JSON layer names do not. Require a unique match.
            pattern = re.compile(
                re.escape(prefix + layer)
                + r"(?:_\d+)?(?:/[^/]+)?/"
                + re.escape(parameter)
                + r":0$"
            )
            found = [v for k, v in self.weights.items() if pattern.fullmatch(k)]
            if len(found) != 1:
                raise ValueError(f"weight not uniquely resolved: {key}")
            self.cache[key] = found[0]
        return self.cache[key]

    def _activation(self, x: Tensor, name: str) -> Tensor:
        if name == "linear":
            return x
        if name == "relu":
            return F.relu(x)
        if name == "sigmoid":
            return x.sigmoid()
        raise ValueError(f"unsupported activation: {name}")

    def _layer(self, layer: dict, inputs: list[Tensor], prefix: str) -> Tensor:
        kind, cfg = layer["class_name"], layer["config"]
        name = cfg["name"]
        x = inputs[0]

        def weight(parameter: str) -> Tensor:
            return self.weight(prefix, name, parameter)

        if kind == "Model":
            return self._run(cfg, inputs, prefix + name + "/")
        if kind == "Lambda":
            function, defaults, closure = cfg["function"]
            digest = hashlib.sha256(function.encode("latin1")).hexdigest()
            if (
                digest not in _LAMBDA_OPERATIONS
                or defaults is not None
                or closure is not None
            ):
                raise ValueError("unreviewed serialized Lambda rejected")
            op = _LAMBDA_OPERATIONS[digest]
            if op == "rgb":
                return (x / 255 - 0.5) * 2
            if op == "triple":
                return x * 3
            if op == "exlu":
                return torch.minimum(
                    torch.exp(x), torch.maximum(torch.ones_like(x), x + 1)
                )
            # Reviewed released arithmetic: interpolation, multiplied by 10.
            index = x * len(self.density) * 0.99999999
            whole = index.to(torch.int64)
            if (whole < 0).any() or (whole + 1 >= len(self.density)).any():
                raise ValueError("density percentile outside released lookup domain")
            fraction = index - whole.to(x.dtype)
            return (
                self.density[whole] * (1 - fraction)
                + self.density[whole + 1] * fraction
            ) * 10
        if kind == "BatchNormalization":
            if cfg["axis"] != -1:
                raise ValueError("only channels-last batch normalization is supported")
            y = (x - weight("moving_mean")) / torch.sqrt(
                weight("moving_variance") + cfg["epsilon"]
            )
            if cfg["scale"]:
                y = y * weight("gamma")
            return y + weight("beta") if cfg["center"] else y
        if kind in ("Conv2D", "SeparableConv2D", "Conv2DTranspose"):
            if cfg["data_format"] != "channels_last":
                raise ValueError("unsupported convolution data format")
            y = x.permute(0, 3, 1, 2)
            stride = tuple(cfg["strides"])
            kernel = tuple(cfg["kernel_size"])
            dilation = tuple(cfg.get("dilation_rate", (1, 1)))
            bias = weight("bias") if cfg["use_bias"] else None
            if kind == "Conv2DTranspose":
                if kernel != (2, 2) or stride != (2, 2) or cfg["padding"] != "same":
                    raise ValueError("unsupported transposed convolution")
                y = F.conv_transpose2d(
                    y, weight("kernel").permute(3, 2, 0, 1), bias, stride
                )
            else:
                if cfg["padding"] == "same":
                    y = tf_same_pad(y, kernel, stride, dilation)
                elif cfg["padding"] != "valid":
                    raise ValueError("unsupported convolution padding")
                if kind == "SeparableConv2D":
                    depth = weight("depthwise_kernel").permute(2, 3, 0, 1)
                    depth = depth.reshape(-1, 1, *kernel)
                    y = F.conv2d(
                        y, depth, stride=stride, dilation=dilation, groups=y.shape[1]
                    )
                    y = F.conv2d(
                        y, weight("pointwise_kernel").permute(3, 2, 0, 1), bias
                    )
                else:
                    y = F.conv2d(
                        y,
                        weight("kernel").permute(3, 2, 0, 1),
                        bias,
                        stride,
                        dilation=dilation,
                    )
            return self._activation(y.permute(0, 2, 3, 1), cfg["activation"])
        if kind == "MaxPooling2D":
            y = x.permute(0, 3, 1, 2)
            if cfg["padding"] == "same":
                y = tf_same_pad(
                    y, tuple(cfg["pool_size"]), tuple(cfg["strides"]), fill=-math.inf
                )
            elif cfg["padding"] != "valid":
                raise ValueError("unsupported pooling padding")
            return F.max_pool2d(y, cfg["pool_size"], cfg["strides"]).permute(0, 2, 3, 1)
        if kind == "Dense":
            y = x @ weight("kernel")
            if cfg["use_bias"]:
                y = y + weight("bias")
            return self._activation(y, cfg["activation"])
        if kind == "Activation":
            return self._activation(x, cfg["activation"])
        if kind in ("Dropout", "GaussianDropout"):
            return x  # Inference, never training.
        if kind == "Reshape":
            return x.reshape(x.shape[0], *cfg["target_shape"])
        if kind == "RepeatVector":
            return x[:, None, :].expand(-1, cfg["n"], -1)
        if kind == "Flatten":
            return x.reshape(x.shape[0], -1)  # Preserve NHWC order.
        if kind == "Concatenate":
            return torch.cat(inputs, dim=cfg["axis"])
        if kind == "Multiply":
            if len(inputs) != 2:
                raise ValueError("unsupported multiply arity")
            return inputs[0] * inputs[1]
        if kind == "Add" or (kind == "Merge" and cfg.get("mode") == "sum"):
            return torch.stack(inputs).sum(0)
        raise ValueError(f"unsupported layer: {kind}")

    def _run(self, config: dict, inputs: list[Tensor], prefix: str) -> Tensor:
        values = {
            name: value for (name, _, _), value in zip(config["input_layers"], inputs)
        }
        if len(values) != len(config["input_layers"]):
            raise ValueError("wrong number of model inputs")
        uses = Counter(
            entry[0]
            for layer in config["layers"]
            for node in layer["inbound_nodes"]
            for entry in node
        )
        output = config["output_layers"]
        if len(output) != 1:
            raise ValueError("unsupported multi-output graph")
        uses[output[0][0]] += 1
        for layer in config["layers"]:
            if layer["class_name"] == "InputLayer":
                continue
            nodes = layer["inbound_nodes"]
            if len(nodes) != 1 or any(entry[2] != 0 for entry in nodes[0]):
                raise ValueError("unsupported shared/multi-output layer")
            names = [entry[0] for entry in nodes[0]]
            values[layer["config"]["name"]] = self._layer(
                layer, [values[n] for n in names], prefix
            )
            for n in names:
                uses[n] -= 1
                if uses[n] == 0:
                    del values[n]
        return values[output[0][0]]

    @torch.inference_mode()
    def __call__(self, inputs: list[np.ndarray]) -> np.ndarray:
        tensors = [
            torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in inputs
        ]
        return self._run(self.config, tensors, "").cpu().numpy()


class Image2MassOffline:
    """Native eight-transform thickness/shape-aware protocol; SI public units."""

    def __init__(self, exports: Path, device: str = "cpu") -> None:
        self.thickness = NumericGraph(exports, "thickness_model", device)
        self.mass = NumericGraph(exports, "shape_aware", device)

    def geometry(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple]:
        image = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        predictions = self.thickness([np.stack([d4(image, i) for i in range(8)])])
        thick = np.mean([d4(predictions[i], i, inverse=True) for i in range(8)], axis=0)
        foreground = (image.sum(axis=2) < 750).astype(np.uint8)
        for operation, size in (
            (cv2.MORPH_CLOSE, (13, 13)),
            (cv2.MORPH_ERODE, (5, 5)),
            (cv2.MORPH_OPEN, (4, 4)),
        ):
            foreground = cv2.morphologyEx(
                foreground,
                operation,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size),
            )
        thick = thick * (
            cv2.resize(foreground, (32, 32), interpolation=cv2.INTER_AREA) > 0.001
        )
        points = np.column_stack(np.where(thick > 0.001)[::-1]).astype(np.float32)
        if len(points) < 3:
            raise ValueError("insufficient foreground for Image2Mass")
        rect = legacy_rectangle(cv2.minAreaRect(points))
        return cv2.resize(thick, (19, 19), interpolation=cv2.INTER_AREA), rect

    def predict(self, rgb: np.ndarray, dimensions_m: np.ndarray) -> dict:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("uint8 RGB image is required")
        if (
            not np.isfinite(dimensions_m).all()
            or np.asarray(dimensions_m).shape != (3,)
            or (np.asarray(dimensions_m) <= 0).any()
        ):
            raise ValueError("three finite positive SI dimensions are required")
        thickness, rect = self.geometry(rgb)
        features = auxiliary_features(
            np.asarray(dimensions_m) / METRES_PER_INCH, thickness, rect
        )
        image = cv2.resize(rgb, (299, 299), interpolation=cv2.INTER_AREA)
        predicted = self.mass(
            [
                np.stack([d4(image, i) for i in range(8)]),
                np.stack([d4(thickness, i) for i in range(8)]),
                np.repeat(features[None], 8, axis=0),
            ]
        ).reshape(-1)
        if (
            len(predicted) != 8
            or not np.isfinite(predicted).all()
            or (predicted <= 0).any()
        ):
            raise ValueError("invalid mass prediction")
        return {
            "mass_kg": float(np.median(predicted)) * KG_PER_POUND,
            "augmented_mass_kg": (predicted * KG_PER_POUND).tolist(),
        }
