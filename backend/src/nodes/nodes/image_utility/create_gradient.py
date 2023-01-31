from __future__ import annotations

import math
from enum import Enum

import numpy as np

from . import category as ImageUtilityCategory
from ...node_base import NodeBase, group
from ...node_factory import NodeFactory
from ...properties import expression
from ...properties.inputs import (
    NumberInput,
    EnumInput, SliderInput, )
from ...properties.outputs import ImageOutput


class ColorMode(Enum):
    RGB = "RGB"
    RGBA = "RGBA"
    GRAYSCALE = "GRAY"


class GradientStyle(Enum):
    HORIZONTAL = "Horizontal"
    VERTICAL = "Vertical"
    DIAGONAL = "Diagonal"
    RADIAL = "Radial"
    CONIC = "Conic"


@NodeFactory.register("chainner:image:create_gradient")
class CreateGradientNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = "Create an image with a gradient."
        self.inputs = [
            NumberInput("Width", minimum=1, unit="px", default=64),
            NumberInput("Height", minimum=1, unit="px", default=64),
            EnumInput(ColorMode, default_value=ColorMode.GRAYSCALE).with_id(2),
            EnumInput(GradientStyle, default_value=GradientStyle.HORIZONTAL).with_id(3),
            SliderInput(
                "Middle Color Position",
                minimum=0,
                maximum=100,
                default=50,
            ),
            group(
                "conditional-enum",
                {
                    "enum": 2,
                    "conditions": [
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGBA.value],
                        [ColorMode.GRAYSCALE.value],
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGBA.value],
                        [ColorMode.GRAYSCALE.value],
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGB.value, ColorMode.RGBA.value],
                        [ColorMode.RGBA.value],
                        [ColorMode.GRAYSCALE.value],
                    ],
                },
            )(
                SliderInput("Red 1", minimum=0, maximum=255, default=0, gradient=["#000000", "#ff0000"]),
                SliderInput("Green 1", minimum=0, maximum=255, default=0, gradient=["#000000", "#00ff00"]),
                SliderInput("Blue 1", minimum=0, maximum=255, default=0, gradient=["#000000", "#0000ff"]),
                SliderInput("Alpha 1", minimum=0, maximum=255, default=0, gradient=["#000000", "#ffffff"]),
                SliderInput("Gray 1", minimum=0, maximum=255, default=0, gradient=["#000000", "#ffffff"]),

                SliderInput("Red 2", minimum=0, maximum=255, default=0, gradient=["#000000", "#ff0000"]),
                SliderInput("Green 2", minimum=0, maximum=255, default=0, gradient=["#000000", "#00ff00"]),
                SliderInput("Blue 2", minimum=0, maximum=255, default=0, gradient=["#000000", "#0000ff"]),
                SliderInput("Alpha 2", minimum=0, maximum=255, default=0, gradient=["#000000", "#ffffff"]),
                SliderInput("Gray 2", minimum=0, maximum=255, default=0, gradient=["#000000", "#ffffff"]),

                SliderInput("Red 3", minimum=0, maximum=255, default=0, gradient=["#000000", "#ff0000"]),
                SliderInput("Green 3", minimum=0, maximum=255, default=0, gradient=["#000000", "#00ff00"]),
                SliderInput("Blue 3", minimum=0, maximum=255, default=0, gradient=["#000000", "#0000ff"]),
                SliderInput("Alpha 3", minimum=0, maximum=255, default=0, gradient=["#000000", "#ffffff"]),
                SliderInput("Gray 3", minimum=0, maximum=255, default=0, gradient=["#000000", "#ffffff"]),
            )
        ]
        self.outputs = [
            ImageOutput(
                image_type=expression.Image(
                    width="Input0",
                    height="Input1",
                )
            )
        ]
        self.category = ImageUtilityCategory
        self.name = "Create Gradient"
        self.icon = "MdFormatColorFill"
        self.sub = "Create Images"

    def _interpolate(self, color1: np.ndarray, color2: np.ndarray, color3: np.ndarray, p: float,
                     middle_position: float):
        if p <= middle_position and middle_position > 0:
            q = p / middle_position
            return color1 * (1 - q) + color2 * q
        else:
            q = (p - middle_position) / (1 - middle_position)
            return color2 * (1 - q) + color3 * q

    def run(
            self, width: int, height: int,
            color_mode: ColorMode,
            gradient_style: GradientStyle,
            middle_position: int,
            red1: int, green1: int, blue1: int, alpha1: int, gray1: int,
            red2: int, green2: int, blue2: int, alpha2: int, gray2: int,
            red3: int, green3: int, blue3: int, alpha3: int, gray3: int,
    ) -> np.ndarray:

        middle_position = middle_position / 100

        img, color1, color2, color3 = None, None, None, None
        if color_mode == ColorMode.RGB:
            img = np.zeros((height, width, 3), dtype=np.float32)
            color1 = np.array([blue1, green1, red1], dtype="float32") / 255
            color2 = np.array([blue2, green2, red2], dtype="float32") / 255
            color3 = np.array([blue3, green3, red3], dtype="float32") / 255
        elif color_mode == ColorMode.RGBA:
            img = np.zeros((height, width, 4), dtype=np.float32)
            color1 = np.array([blue1, green1, red1, alpha1], dtype="float32") / 255
            color2 = np.array([blue2, green2, red2, alpha2], dtype="float32") / 255
            color3 = np.array([blue3, green3, red3, alpha3], dtype="float32") / 255
        elif color_mode == ColorMode.GRAYSCALE:
            img = np.zeros((height, width, 1), dtype=np.float32)
            color1 = np.array([gray1], dtype="float32") / 255
            color2 = np.array([gray2], dtype="float32") / 255
            color3 = np.array([gray3], dtype="float32") / 255

        # TODO vectorize these...  too slow
        if gradient_style == GradientStyle.HORIZONTAL:
            if width == 1:
                raise RuntimeError("Horizontal gradient needs at least two width.")
            for column in range(width):
                p = column / (width - 1)
                img[:, column] = self._interpolate(color1, color2, color3, p, middle_position)

        elif gradient_style == GradientStyle.VERTICAL:
            if height == 1:
                raise RuntimeError("Vertical gradient needs at least two height.")
            for row in range(height):
                p = row / (height - 1)
                img[row, :] = self._interpolate(color1, color2, color3, p, middle_position)

        elif gradient_style == GradientStyle.DIAGONAL:
            diagonal = np.array([width, height], dtype="float32")
            diagonal_length = np.sqrt(np.sum(diagonal ** 2))
            diagonal /= diagonal_length
            for column in range(width):
                for row in range(height):
                    projection = diagonal.dot(np.array([column, row]))
                    length = np.sqrt(np.sum(projection ** 2))
                    p = length / (diagonal_length - 1)
                    img[row, column] = self._interpolate(color1, color2, color3, p, middle_position)

        elif gradient_style == GradientStyle.RADIAL:

            inner_radius = 0
            outer_radius = width / 2

            center = np.array([width, height], dtype="float32") / 2
            for column in range(width):
                for row in range(height):
                    distance = np.sqrt(np.sum((np.array([column, row]) - center) ** 2))
                    if distance <= inner_radius:
                        color = color1
                    elif distance >= outer_radius:
                        color = color3
                    else:
                        p = (distance - inner_radius) / (outer_radius - inner_radius)
                        color = self._interpolate(color1, color2, color3, p, middle_position)
                    img[row, column] = color

        elif gradient_style == GradientStyle.CONIC:
            # TODO rotation parameter
            center = np.array([width, height]) / 2
            for column in range(width):
                for row in range(height):
                    delta = np.array([column, row]) - center
                    angle = math.atan2(delta[1], delta[0])
                    if angle < 0:
                        angle += np.pi * 2
                    p = angle / np.pi / 2
                    img[row, column] = self._interpolate(color1, color2, color3, p, middle_position)

        return img
