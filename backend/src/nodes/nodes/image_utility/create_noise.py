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
    FRACTAL_SIMPLEX = "Simplex (3 Octave)"
    VORONOI = "Voronoi"


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
            EnumInput(NoiseMethod, default_value=NoiseMethod.SIMPLEX).with_id(3),
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
                NumberInput("Frequency Scale", minimum=1, default=1, precision=1).with_id(4),
                SliderInput("Max Output (%)", minimum=0, default=100, maximum=100, precision=2).with_id(5),
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

    def _add_simplex(self, image: np.ndarray, seed: int, scale: float, brightness: float):
        sn = SimplexNoise(2)

        points = np.array([(i, j) for i in range(image.shape[0]) for j in range(image.shape[1])])
        output = sn.evaluate(points / scale, seed=seed)

        for (i, j), v in zip(points, output):
            image[i, j] += v * brightness

    def _add_simplex_tiled(self, image: np.ndarray, seed: int, scale: float, brightness: float):
        sn = SimplexNoise(4)

        pixels = np.array([(i, j) for i in range(image.shape[0]) for j in range(image.shape[1])])
        points = np.stack([
            image.shape[0] * (np.sin(pixels[:, 0] * 2 * np.pi / image.shape[0]) + 1),
            image.shape[0] * (np.cos(pixels[:, 0] * 2 * np.pi / image.shape[0]) + 1),
            image.shape[1] * (np.sin(pixels[:, 1] * 2 * np.pi / image.shape[1]) + 1),
            image.shape[1] * (np.cos(pixels[:, 1] * 2 * np.pi / image.shape[1]) + 1),
        ], axis=1)
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
            self._add_simplex_tiled(img, seed, scale, brightness)
            return np.clip(img, 0, 1)

        if noise_method == NoiseMethod.FRACTAL_SIMPLEX:
            img = np.zeros((height, width), dtype="float32")
            self._add_simplex(img, seed, scale, brightness / 2)
            self._add_simplex(img, seed, scale / 2, brightness / 4)
            self._add_simplex(img, seed, scale / 4, brightness / 8)
            return np.clip(img, 0, 1)
