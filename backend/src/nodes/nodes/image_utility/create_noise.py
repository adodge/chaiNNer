from __future__ import annotations

from enum import Enum

import numpy as np

from . import category as ImageUtilityCategory
from ...impl.noise_functions.simplex import SimplexNoise
from ...impl.noise_functions.value import ValueNoise
from ...node_base import NodeBase, group
from ...node_factory import NodeFactory
from ...properties import expression
from ...properties.inputs import (
    NumberInput,
    EnumInput, SliderInput, BoolInput, ImageInput,
)
from ...properties.outputs import ImageOutput


class NoiseMethod(Enum):
    VALUE = "Value Noise"
    SIMPLEX = "Simplex"


class FractalMethod(Enum):
    NONE = "None"
    PINK = "Pink noise"


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
            EnumInput(
                NoiseMethod,
                default_value=NoiseMethod.SIMPLEX,
                option_labels={key: key.value for key in NoiseMethod}
            ).with_id(3),
            NumberInput("Scale", minimum=1, default=1, precision=1).with_id(4),
            SliderInput("Brightness", minimum=0, default=100, maximum=100, precision=2).with_id(5),
            BoolInput("Tile Horizontal", default=False).with_id(10),
            BoolInput("Tile Vertical", default=False).with_id(11),
            EnumInput(
                FractalMethod,
                default_value=FractalMethod.NONE,
                option_labels={key: key.value for key in FractalMethod}
            ).with_id(6),
            group(
                "conditional-enum",
                {
                    "enum": 6,
                    "conditions": [FractalMethod.PINK.value, FractalMethod.PINK.value, FractalMethod.PINK.value,
                                   FractalMethod.PINK.value],
                },
            )(
                NumberInput("Layers", minimum=2, default=3, precision=1).with_id(7),
                NumberInput("Scale Ratio", minimum=1, default=2, precision=2).with_id(8),
                NumberInput("Brightness Ratio", minimum=1, default=2, precision=2).with_id(9),
                BoolInput("Increment Seed", default=False).with_id(12),
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

    def _add_noise(self, generator_class, image: np.ndarray, scale: float, brightness: float,
                   tile_horizontal: bool = False, tile_vertical: bool = False, **kwargs):
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

        gen = generator_class(dimensions=points.shape[1], **kwargs)
        output = gen.evaluate(points / scale)

        for (i, j), v in zip(pixels, output):
            image[i, j] += v * brightness

    def run(
            self, width: int, height: int, seed: int, noise_method: NoiseMethod,
            scale: float, brightness: float, tile_horizontal: bool, tile_vertical: bool, fractal_method: FractalMethod,
            layers: int, scale_ratio: float, brightness_ratio: float,
            increment_seed: bool,
    ) -> np.ndarray:
        img = np.zeros((height, width), dtype="float32")
        brightness /= 100

        kwargs = {
            'tile_horizontal': tile_horizontal,
            'tile_vertical': tile_vertical,
            'scale': scale,
            'brightness': brightness,
            'seed': seed,
        }

        generator_class = None
        if noise_method == NoiseMethod.SIMPLEX:
            generator_class = SimplexNoise
        elif noise_method == NoiseMethod.VALUE:
            generator_class = ValueNoise

        if fractal_method == FractalMethod.NONE:
            self._add_noise(generator_class, image=img, **kwargs)
        elif fractal_method == FractalMethod.PINK:
            del kwargs['scale'], kwargs['brightness']
            total_brightness = 0
            relative_brightness = 1
            for i in range(layers):
                total_brightness += relative_brightness
                self._add_noise(generator_class, image=img, **kwargs, scale=scale,
                                brightness=brightness * relative_brightness)
                scale /= scale_ratio
                relative_brightness /= brightness_ratio
                if increment_seed:
                    kwargs['seed'] = (kwargs['seed'] + 1) % (2 ** 32)
            img /= total_brightness

        return np.clip(img, 0, 1)


class EffectType(Enum):
    TURBULENCE="turbulence"
    WOOD_GRAIN="wood grain"
    MARBLE="marble"


@NodeFactory.register("chainner:image:noise_effect")
class NoiseEffect(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Apply an effect."
        self.inputs = [
            ImageInput(),
            EnumInput(
                EffectType,
                default_value=EffectType.TURBULENCE,
                option_labels={key: key.value for key in EffectType}
            ).with_id(3),
            NumberInput("Scale", minimum=1, default=1, precision=1).with_id(4),
            SliderInput("Brightness", minimum=0, default=100, maximum=100, precision=2).with_id(5),
            BoolInput("Tile Horizontal", default=False).with_id(10),
            BoolInput("Tile Vertical", default=False).with_id(11),
            EnumInput(
                FractalMethod,
                default_value=FractalMethod.NONE,
                option_labels={key: key.value for key in FractalMethod}
            ).with_id(6),
            group(
                "conditional-enum",
                {
                    "enum": 6,
                    "conditions": [FractalMethod.PINK.value, FractalMethod.PINK.value, FractalMethod.PINK.value,
                                   FractalMethod.PINK.value],
                },
            )(
                NumberInput("Layers", minimum=2, default=3, precision=1).with_id(7),
                NumberInput("Scale Ratio", minimum=1, default=2, precision=2).with_id(8),
                NumberInput("Brightness Ratio", minimum=1, default=2, precision=2).with_id(9),
                BoolInput("Increment Seed", default=False).with_id(12),
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

# TODO
# Perlin noise
# Voronoi noise
# Grids
# Effects: Turbulence, Marble, Wood grain
# https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/procedural-patterns-noise-part-1/simple-pattern-examples.html
# Vector field from perlin noise
# Warp an image with a vector field
# skew
# histograms
# UV mapping