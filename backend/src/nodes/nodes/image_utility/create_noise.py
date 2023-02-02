from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np
from sanic.log import logger

from . import category as ImageUtilityCategory
from ...impl.image_utils import as_3d, cartesian_product
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
        pixels = cartesian_product([np.arange(image.shape[0]), np.arange(image.shape[1])])
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

        image += output.reshape(image.shape) * brightness

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
    # https://dl.acm.org/doi/10.1145/325165.325247
    def __init__(self):
        super().__init__()
        self.description = "Apply an effect."
        self.inputs = [
            ImageInput(),
            EnumInput(
                EffectType,
                default_value=EffectType.TURBULENCE,
                option_labels={key: key.value for key in EffectType}
            ).with_id(1),
            group(
                "conditional-enum",
                {
                    "enum": 1,
                    "conditions": [EffectType.WOOD_GRAIN.value, EffectType.MARBLE.value, EffectType.MARBLE.value],
                },
            )(
                NumberInput("Ring Count", minimum=2, default=2),
                NumberInput("Bands", minimum=1, default=10),
                NumberInput("Warp Amount", default=5, precision=1),
            ),
        ]
        self.outputs = [
            ImageOutput(
                image_type=expression.Image(size_as="Input0", channels_as="Input0"),
            )
        ]
        self.category = ImageUtilityCategory
        self.name = "Noise Effect"
        self.icon = "MdFormatColorFill"
        self.sub = "Noise Effect"

    def run(self, image: np.ndarray, effect_type: EffectType, ring_count: int, bands: float, warp_amount: float):
        if effect_type == EffectType.TURBULENCE:
            return np.abs(image-0.5)*2
        elif effect_type == EffectType.WOOD_GRAIN:
            return np.remainder(image*ring_count, 1)
            # TODO solid regions, invert odd regions
        elif effect_type == EffectType.MARBLE:
            image = as_3d(image)
            x = np.arange(image.shape[1]).reshape((1,-1,1)) / image.shape[1] * np.pi * 2 * bands
            return (np.sin((x + (image*2-1) * warp_amount)) + 1)/2


@NodeFactory.register("chainner:image:extract_histogram")
class IntensityHistogram(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Count the proportion of different intensities in the input image and output a histogram. " \
                           "(1 row image where pixel intensity is proportional to count) "
        self.inputs = [
            ImageInput(),
            NumberInput("Number of Bins", minimum=1, default=256),
        ]
        self.outputs = [
            ImageOutput(
                image_type=expression.Image(width="Input1", height=1, channels_as="Input0"),
            )
        ]
        self.category = ImageUtilityCategory
        self.name = "Measure Intensity"
        self.icon = "MdFormatColorFill"
        self.sub = "Noise Effect"

    def run(self, image: np.ndarray, n_bins: int):
        image = as_3d(image)
        output = np.zeros((1, n_bins, image.shape[2]), dtype='float32')
        binned = np.floor(image*n_bins).astype("int32")
        for ch in range(image.shape[2]):
            unique, counts = np.unique(binned[:,:,ch], return_counts=True)
            for bin,count in zip(unique, counts):
                if bin == n_bins:
                    output[0, bin-1, ch] += count
                else:
                    output[0, bin, ch] += count
            output[:,:,ch] /= np.max(output[:,:,ch])
        return output


@NodeFactory.register("chainner:image:draw_curve")
class DrawCurve(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Take a 1 row image and plot a curve (for each channel) where the height of the curve is guided by the intensity of the input."
        self.inputs = [
            ImageInput(),
            NumberInput("Height", minimum=1, unit="px", default=1),
        ]
        self.outputs = [
            ImageOutput(
                image_type=expression.Image(width_as="Input0", height="Input1", channels_as="Input0"),
            )
        ]
        self.category = ImageUtilityCategory
        self.name = "Draw Curve"
        self.icon = "MdFormatColorFill"
        self.sub = "Noise Effect"

    def run(self, image: np.ndarray, height: int):
        image = as_3d(image)
        output = np.zeros((height, image.shape[1], image.shape[2]))
        for ch in range(image.shape[2]):
            scale = np.max(image[:,:,ch])
            for column in range(image.shape[1]):
                y = int(np.floor(height * image[0,column,ch] / scale))
                output[(height-y-1):, column, ch] = 1
        return output


def dtype_to_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.dtype("float32"):
        return image
    max_value = np.iinfo(image.dtype).max
    return image.astype(np.dtype("float32")) / max_value


@NodeFactory.register("chainner:image:create_vector_field")
class NoiseVectorField(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Take a greyscale image and interpret it as a field of angles.  Returns a normal map with normalized vectors in the red and green channels.  Black pixels will be mapped to 'Start angle' and white pixels will be mapped to 'Start angle + Angle Range'"
        self.inputs = [
            ImageInput(channels=1),
            NumberInput("Start Angle", minimum=0, maximum=360, unit="degree", default=0),
            NumberInput("Angle Range", minimum=0, maximum=360, unit="degree", default=360),
        ]
        self.outputs = [
            ImageOutput(
                image_type=expression.Image(size_as="Input0", channels=3),
            )
        ]
        self.category = ImageUtilityCategory
        self.name = "Vector Field"
        self.icon = "MdFormatColorFill"
        self.sub = "Noise Effect"

    def run(self, image: np.ndarray, angle0: float, d_angle: float):
        image = dtype_to_float(image)
        if image.ndim == 3:
            assert image.shape[2] == 1
            image = image.reshape((image.shape[:2]))
        radians = (image*d_angle + angle0) * np.pi / 180
        red, green, blue = np.cos(radians)/2+1/2, np.sin(radians)/2+1/2, np.zeros_like(image)
        return np.stack([blue,green,red], axis=2)


@NodeFactory.register("chainner:image:simulate_particle_movement")
class SimulateParticles(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Simulate the movement of a bunch of particles on a vector field."
        self.inputs = [
            ImageInput(channels=3),
            group("seed")(
                NumberInput("Seed", minimum=0, maximum=2 ** 32 - 1, default=0),
            ),
            NumberInput("Number of Particles", minimum=0, default=100),
            NumberInput("Simulation Steps", minimum=0, default=100),
            BoolInput("Wrap Horizontal", default=False),
            BoolInput("Wrap Vertical", default=False),
        ]
        self.outputs = [
            ImageOutput(
                image_type=expression.Image(size_as="Input0", channels=1),
            ),
            ImageOutput(
                image_type=expression.Image(size_as="Input0", channels=1),
            )
        ]
        self.category = ImageUtilityCategory
        self.name = "Particle Simulation"
        self.icon = "MdFormatColorFill"
        self.sub = "Noise Effect"

    def run(self, image: np.ndarray, seed: int, n_particles:int, n_steps: int, wrap_horizontal: bool, wrap_vertical: bool) -> Tuple[np.ndarray, np.ndarray]:
        flat_image = image[:,:,1:3].reshape((-1,2))

        np.random.seed(seed)
        position = np.random.random((n_particles, 2))*np.array(image.shape[:2]).reshape((1,-1))
        trail = np.zeros((flat_image.shape[0]), dtype=np.int32)

        for _ in range(n_steps):
            pixel = np.floor(position).astype(np.int32)
            indices = pixel[:,0] + pixel[:,1]*image.shape[0]
            np.add.at(trail, indices, 1)
            velocity = flat_image[indices]*2-1
            position += velocity

            if wrap_vertical:
                position[position[:,0] > image.shape[0], 0] -= image.shape[0]
                position[position[:,0] < 0, 0] += image.shape[0]

            if wrap_horizontal:
                position[position[:,1] > image.shape[1], 1] -= image.shape[1]
                position[position[:,1] < 0, 1] += image.shape[1]

            out_of_bounds = np.any([(position[:,0] > image.shape[0]), (position[:,0] < 0), (position[:,1] > image.shape[1]), (position[:,1] < 0)], axis=0)
            num_out_of_bounds = out_of_bounds.sum()
            if num_out_of_bounds:
                position[out_of_bounds] = np.random.random((num_out_of_bounds, 2))*np.array(image.shape[:2]).reshape((1,-1))

        pixel = np.floor(position).astype(np.int32)
        indices = pixel[:,0] + pixel[:,1]*image.shape[0]
        final = np.zeros((flat_image.shape[0]), dtype=np.int32)
        np.add.at(final, indices, 1)

        trail_image = trail.reshape(image.shape[:2]).astype(np.float32) / np.max(trail)
        final_image = final.reshape(image.shape[:2]).astype(np.float32) / np.max(final)

        return final_image, trail_image

# TODO
# Perlin noise
# Voronoi noise
# musgrave noise
# Vector field from perlin noise
# Warp an image with a vector field
# affine transform
# map transforms (polar-cartesian)
# procurstes analysis
# UV mapping
# background remover
# img2depth