from __future__ import annotations

from enum import Enum

import numpy as np

from . import category as ImageUtilityCategory
from ...impl.simplex import SimplexNoise
from ...node_base import NodeBase, group
from ...node_factory import NodeFactory
from ...properties import expression
from ...properties.inputs import (
    NumberInput,
    EnumInput, SliderInput,
)
from ...properties.outputs import ImageOutput


class NoiseMethod(Enum):
    PERLIN = "Perlin"
    SIMPLEX = "Simplex"
    SIMPLEX_HTILE = "Simplex (tiled horizontal)"
    SIMPLEX_VTILE = "Simplex (tiled vertically)"
    SIMPLEX_ATILE = "Simplex (tiled both)"
    FRACTAL_SIMPLEX = "Simplex (3 Octave)"
    VORONOI = "Voronoi"


NOISE_METHOD_LABELS = {key: key.value for key in NoiseMethod}


@NodeFactory.register("chainner:image:create_noise")
class CreateNoiseNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Create an image of specified dimensions filled with one of a variety of noises."
        self.inputs = [
            NumberInput("Width", minimum=1, unit="px", default=1),
            NumberInput("Height", minimum=1, unit="px", default=1),
            group("seed")(
                NumberInput("Seed", minimum=0, maximum=2 ** 32 - 1, default=0),
            ),
            EnumInput(NoiseMethod, default_value=NoiseMethod.SIMPLEX, option_labels=NOISE_METHOD_LABELS).with_id(3),
            group(
                "conditional-enum",
                {
                    "enum": 3,
                    "conditions": [
                        [NoiseMethod.PERLIN.value, NoiseMethod.SIMPLEX.value, NoiseMethod.FRACTAL_SIMPLEX.value],
                        [NoiseMethod.PERLIN.value, NoiseMethod.SIMPLEX.value, NoiseMethod.FRACTAL_SIMPLEX.value],
                    ],
                },
            )(
                NumberInput("Scale", minimum=1, default=1, precision=1).with_id(4),
                SliderInput("Brightness", minimum=0, default=100, maximum=100, precision=2).with_id(5),
            ),
        ]
        self.outputs = [
            ImageOutput(
                image_type=expression.Image(
                    width="Input0",
                    height="Input1",
                    channels="1",
                )
            )
        ]
        self.category = ImageUtilityCategory
        self.name = "Create Noise"
        self.icon = "MdFormatColorFill"
        self.sub = "Create Images"

    @staticmethod
    def _add_simplex(image: np.ndarray, seed: int, scale: float, brightness: float,
                     tile_horizontal: bool = False, tile_vertical: bool = False):
        pixels = np.array([(i, j) for i in range(image.shape[0]) for j in range(image.shape[1])])
        points = np.array(pixels)
        if tile_horizontal:
            x = points[:, 1] * 2 * np.pi / image.shape[1]
            cx = (image.shape[1] * np.cos(x) / np.pi / 2).reshape((-1, 1))
            sx = (image.shape[1] * np.sin(x) / np.pi / 2).reshape((-1, 1))
            points = np.concatenate([points[:, :1], cx, sx], axis=1)
        if tile_vertical:
            x = points[:, 0] * 2 * np.pi / image.shape[0]
            cx = (image.shape[0] * np.cos(x) / np.pi / 2).reshape((-1, 1))
            sx = (image.shape[0] * np.sin(x) / np.pi / 2).reshape((-1, 1))
            points = np.concatenate([points[:, 1:], cx, sx], axis=1)

        sn = SimplexNoise(points.shape[1])
        output = sn.evaluate(points / scale, seed=seed)

        for (i, j), v in zip(pixels, output):
            image[i, j] += v * brightness

    def run(
            self, width: int, height: int, seed: int, noise_method: NoiseMethod,
            scale: float, brightness: float
    ) -> np.ndarray:
        brightness = brightness / 100

        if noise_method == NoiseMethod.SIMPLEX:
            img = np.zeros((height, width), dtype="float32")
            self._add_simplex(img, seed, scale, brightness)
            return np.clip(img, 0, 1)

        if noise_method == NoiseMethod.SIMPLEX_HTILE:
            img = np.zeros((height, width), dtype="float32")
            self._add_simplex(img, seed, scale, brightness, tile_horizontal=True)
            return np.clip(img, 0, 1)
        if noise_method == NoiseMethod.SIMPLEX_VTILE:
            img = np.zeros((height, width), dtype="float32")
            self._add_simplex(img, seed, scale, brightness, tile_vertical=True)
            return np.clip(img, 0, 1)
        if noise_method == NoiseMethod.SIMPLEX_ATILE:
            img = np.zeros((height, width), dtype="float32")
            self._add_simplex(img, seed, scale, brightness, tile_horizontal=True, tile_vertical=True)
            return np.clip(img, 0, 1)

        if noise_method == NoiseMethod.FRACTAL_SIMPLEX:
            img = np.zeros((height, width), dtype="float32")
            self._add_simplex(img, seed, scale, brightness / 2)
            self._add_simplex(img, seed, scale / 2, brightness / 4)
            self._add_simplex(img, seed, scale / 4, brightness / 8)
            return np.clip(img, 0, 1)
